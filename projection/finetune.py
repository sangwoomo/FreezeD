import os, sys, time
import shutil
import yaml
import random
import numpy as np
import cupy
from copy import deepcopy

import argparse
import chainer
from chainer import training
from chainer.training import extension
from chainer.training import extensions

sys.path.append(os.path.dirname(__file__))

from evaluation import sample_generate, sample_generate_conditional, sample_generate_light, calc_inception, calc_FID
import source.yaml_utils as yaml_utils


def create_result_dir(result_dir, config_path, config):
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    def copy_to_result_dir(fn, result_dir):
        bfn = os.path.basename(fn)
        shutil.copy(fn, '{}/{}'.format(result_dir, bfn))

    copy_to_result_dir(config_path, result_dir)
    copy_to_result_dir(config.models['generator']['fn'], result_dir)
    copy_to_result_dir(config.models['discriminator']['fn'], result_dir)
    copy_to_result_dir(config.dataset['dataset_fn'], result_dir)
    copy_to_result_dir(config.updater['fn'], result_dir)


def load_models(config, mode):
    gen_conf = deepcopy(config.models['generator'])
    dis_conf = deepcopy(config.models['discriminator'])

    if mode == 'source':
        gen_conf['args']['n_classes'] = gen_conf['args']['n_classes_src']
        dis_conf['args']['n_classes'] = dis_conf['args']['n_classes_src']
    elif mode == 'target':
        gen_conf['args']['n_classes'] = gen_conf['args']['n_classes_tgt']
        dis_conf['args']['n_classes'] = dis_conf['args']['n_classes_tgt']
    else:
        raise NotImplementedError

    gen_conf['args'].pop('n_classes_src')
    gen_conf['args'].pop('n_classes_tgt')
    dis_conf['args'].pop('n_classes_src')
    dis_conf['args'].pop('n_classes_tgt')

    gen = yaml_utils.load_model(gen_conf['fn'], gen_conf['name'], gen_conf['args'])
    dis = yaml_utils.load_model(dis_conf['fn'], dis_conf['name'], dis_conf['args'])

    return gen, dis


# https://chainercv.readthedocs.io/en/stable/tutorial/link.html
def get_shape_mismatch_names(src, dst):
    # all parameters are assumed to be initialized
    mismatch_names = []
    src_params = {p[0]: p[1] for p in src.namedparams()}
    for dst_named_param in dst.namedparams():
        name = dst_named_param[0]
        dst_param = dst_named_param[1]
        src_param = src_params[name]
        if src_param.shape != dst_param.shape:
            mismatch_names.append(name)
    return mismatch_names


# https://chainercv.readthedocs.io/en/stable/tutorial/link.html
def load_parameters(src, dst):
    # all parameters are assumed to be initialized
    ignore_names = get_shape_mismatch_names(src, dst)
    src_params = {p[0]: p[1] for p in src.namedparams()}
    for dst_named_param in dst.namedparams():
        name = dst_named_param[0]
        if name not in ignore_names:
            dst_named_param[1].array[:] = src_params[name].array[:]


def make_optimizer(model, alpha=0.0002, beta1=0., beta2=0.9):
    optimizer = chainer.optimizers.Adam(alpha=alpha, beta1=beta1, beta2=beta2)
    optimizer.setup(model)
    return optimizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='configs/base.yml', help='path to config file')
    parser.add_argument('--gpu', type=int, default=0, help='index of gpu to be used')
    parser.add_argument('--data_dir', type=str, default='./dataset/imagenet')
    parser.add_argument('--results_dir', type=str, default='./results/temp', help='directory to save the results to')
    parser.add_argument('--loaderjob', type=int, help='number of parallel data loading processes')
    parser.add_argument('--layer', type=int, default=0, help='freeze discriminator layer')

    args = parser.parse_args()
    config = yaml_utils.Config(yaml.load(open(args.config_path), Loader=yaml.FullLoader))
    chainer.cuda.get_device_from_id(args.gpu).use()

    # Fix randomness
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    cupy.random.seed(config['seed'])

    # Model
    G_src, D_src = load_models(config, mode='source')
    G_tgt, D_tgt = load_models(config, mode='target')

    G_src.to_gpu(device=args.gpu)
    D_src.to_gpu(device=args.gpu)
    G_tgt.to_gpu(device=args.gpu)
    D_tgt.to_gpu(device=args.gpu)

    models = {"G_src": G_src, "D_src": D_src, "G_tgt": G_tgt, "D_tgt": D_tgt}

    # Optimizer
    opt_G_tgt = make_optimizer(G_tgt, alpha=config.adam['alpha'], beta1=config.adam['beta1'], beta2=config.adam['beta2'])
    opt_D_tgt = make_optimizer(D_tgt, alpha=config.adam['alpha'], beta1=config.adam['beta1'], beta2=config.adam['beta2'])

    for i in range(1, args.layer + 1):  # freeze discriminator
        getattr(D_tgt, f'block{i}').disable_update()

    opts = {"opt_G_tgt": opt_G_tgt, "opt_D_tgt": opt_D_tgt}

    # Dataset
    config['dataset']['args']['root'] = args.data_dir
    dataset = yaml_utils.load_dataset(config)

    # Iterator
    iterator = chainer.iterators.MultiprocessIterator(dataset, config.batchsize, n_processes=args.loaderjob)
    kwargs = config.updater['args'] if 'args' in config.updater else {}
    kwargs.update({
        'models': models,
        'iterator': iterator,
        'optimizer': opts,
    })
    updater = yaml_utils.load_updater_class(config)
    updater = updater(**kwargs)
    out = args.results_dir
    create_result_dir(out, args.config_path, config)
    trainer = training.Trainer(updater, (config.iteration, 'iteration'), out=out)
    report_keys = ["loss_dis", "loss_gen", "loss_FM", "FID"]

    # Set up logging
    trainer.extend(extensions.snapshot(filename='snapshot_best'),
                   trigger=training.triggers.MinValueTrigger("FID", trigger=(config.evaluation_interval, 'iteration')))
    for m in models.values():
        trainer.extend(extensions.snapshot_object(m, m.__class__.__name__ + '_best.npz'),
                       trigger=training.triggers.MinValueTrigger("FID", trigger=(config.evaluation_interval, 'iteration')))

    trainer.extend(extensions.LogReport(keys=report_keys, trigger=(config.display_interval, 'iteration')))
    trainer.extend(extensions.PrintReport(report_keys), trigger=(config.display_interval, 'iteration'))

    trainer.extend(sample_generate_conditional(G_tgt, out, n_classes=G_tgt.n_classes),
                   trigger=(config.evaluation_interval, 'iteration'),
                   priority=extension.PRIORITY_WRITER)
    trainer.extend(sample_generate_light(G_tgt, out, rows=10, cols=10),
                   trigger=(config.evaluation_interval // 10, 'iteration'),
                   priority=extension.PRIORITY_WRITER)
    trainer.extend(calc_FID(G_tgt, n_ims=5000, stat_file=config['eval']['stat_file']),
                   trigger=(config.evaluation_interval, 'iteration'),
                   priority=extension.PRIORITY_WRITER)
    trainer.extend(extensions.ProgressBar(update_interval=config.progressbar_interval))

    ext_opt_gen = extensions.LinearShift('alpha', (config.adam['alpha'], 0.), (config.iteration_decay_start, config.iteration), opt_G_tgt)
    ext_opt_dis = extensions.LinearShift('alpha', (config.adam['alpha'], 0.), (config.iteration_decay_start, config.iteration), opt_D_tgt)
    trainer.extend(ext_opt_gen)
    trainer.extend(ext_opt_dis)

    # Load source networks
    chainer.serializers.load_npz(config['pretrained']['gen'], trainer.updater.models['G_src'])
    chainer.serializers.load_npz(config['pretrained']['dis'], trainer.updater.models['D_src'])
    load_parameters(trainer.updater.models['G_src'], trainer.updater.models['G_tgt'])
    load_parameters(trainer.updater.models['D_src'], trainer.updater.models['D_tgt'])

    # Run the training
    print("start training")
    trainer.run()


if __name__ == '__main__':
    main()
