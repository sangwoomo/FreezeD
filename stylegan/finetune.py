import os
import argparse
import pickle
import math
import random
import numpy as np
from tqdm import tqdm

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import grad
from torchvision import transforms, utils
from torch.utils.tensorboard import SummaryWriter

from dataset import MultiResolutionDataset
from model import StyledGenerator, Discriminator, Miner
from train import accumulate, sample_data, adjust_lr
from metric.inception import InceptionV3
from metric.metric import get_fake_images_and_acts, compute_fid
from loss.AdaBIGGANLoss import AdaBIGGANLoss


# override requires_grad function
def requires_grad(model, flag=True, target_layer=None):
	for name, param in model.named_parameters():
		if target_layer is None:  # every layer
			param.requires_grad = flag
		elif target_layer in name:  # target layer
			param.requires_grad = flag


def evaluate(iteration):
	"""Custom GAN evaluation function"""

	# save images

	gen_i, gen_j = args.gen_sample.get(args.image_size, (10, 5))

	images = []
	with torch.no_grad():
		for i in range(gen_i):
			images.append(G_running_target(fixed_noise[i].cuda(), step=step, alpha=alpha).cpu())

	sample_path = f'sample/{args.name}/{str(iteration).zfill(6)}.png'
	utils.save_image(torch.cat(images, dim=0), sample_path, nrow=gen_i, normalize=True, range=(-1, 1))

	# compute evaluation metrics

	sample_num = args.sample_num
	fake_images, fake_acts = get_fake_images_and_acts(inception, G_running_target, code_size, step, alpha, sample_num, batch_size)

	fid = compute_fid(real_acts, fake_acts)

	metrics = {'fid': fid}

	return metrics


def l2_reg(net_src, net_tgt):
	params_src = list(net_src.parameters())
	params_tgt = list(net_tgt.parameters())

	loss = 0
	for p_src, p_tgt in zip(params_src, params_tgt):
		loss += F.mse_loss(p_tgt, p_src)

	return loss


def FM_reg(real_image, feature_loc):
	feat_src = D_source(real_image, step=step, alpha=alpha, get_feature=True, feature_loc=feature_loc)
	feat_tgt = D_target(real_image, step=step, alpha=alpha, get_feature=True, feature_loc=feature_loc)

	return F.mse_loss(feat_tgt, feat_src)


def finetune(args, dataset, G_target, D_target):
	### create logger ###

	if not os.path.exists(f'checkpoint/{args.name}'):
		os.makedirs(f'checkpoint/{args.name}')

	if not os.path.exists(f'sample/{args.name}'):
		os.makedirs(f'sample/{args.name}')

	logger = SummaryWriter(f'checkpoint/{args.name}')

	### initialize experiment ###

	global step  # simple hack
	global batch_size  # simple hack

	step = int(math.log2(args.image_size)) - 2
	resolution = 4 * 2 ** step

	batch_size = args.batch.get(resolution, args.batch_default) * torch.cuda.device_count()  # DataParallel
	loader = sample_data(dataset, batch_size, resolution)
	loader_iter = iter(loader)

	### run experiment ###

	pbar = tqdm(range(args.phase), position=0)

	metrics = evaluate(iteration=0)
	for key, val in metrics.items():
		logger.add_scalar(key, val, 0)

	best_fid = metrics['fid']

	for i in pbar:
		adjust_lr(G_optimizer, args.lr.get(resolution, 0.001))
		adjust_lr(D_optimizer, args.lr.get(resolution, 0.001))

		### sample data and noise ###

		try:
			real_index, real_image = next(loader_iter)

		except:
			loader_iter = iter(loader)
			real_index, real_image = next(loader_iter)

		gen_in1, gen_in2 = sample_noise(len(real_image))

		### update D ###

		D_target.zero_grad()

		requires_grad(G_target, False)
		if args.freeze_D:
			for loc in range(args.feature_loc):
				requires_grad(D_target, True, target_layer=f'progression.{8 - loc}')
			requires_grad(D_target, True, target_layer=f'linear')
		else:
			requires_grad(D_target, True)

		D_loss_val, grad_loss_val = backward_D(args, G_target, D_target, real_image, gen_in1)

		D_optimizer.step()

		### update G ###

		G_target.zero_grad()

		if not args.miner:
			requires_grad(G_target, True)  # do not update G
		if args.freeze_D:
			for loc in range(args.feature_loc):
				requires_grad(D_target, False, target_layer=f'progression.{8 - loc}')
			requires_grad(D_target, False, target_layer=f'linear')
		else:
			requires_grad(D_target, False)

		G_loss_val = backward_G(args, G_target, D_target, gen_in2)

		G_optimizer.step()
		accumulate(G_running_target, G_target.module)

		### save results and checkpoints ###

		if (i + 1) % args.eval_step == 0:
			logger.add_scalar('G_loss_val', G_loss_val, i + 1)
			logger.add_scalar('D_loss_val', D_loss_val, i + 1)
			logger.add_scalar('grad_loss_val', grad_loss_val, i + 1)

			metrics = evaluate(iteration=i + 1)
			for key, val in metrics.items():
				logger.add_scalar(key, val, i + 1)

			if metrics['fid'] < best_fid:
				torch.save({
					'generator': G_target.module.state_dict(),
					'discriminator': D_target.module.state_dict(),
					'g_optimizer': G_optimizer.state_dict(),
					'd_optimizer': D_optimizer.state_dict(),
					'g_running': G_running_target.state_dict(),
				}, f'checkpoint/{args.name}/best.model')

		state_msg = f'Size: {4 * 2 ** step}; G: {G_loss_val:.3f}; D: {D_loss_val:.3f}; Grad: {grad_loss_val:.3f};'
		if metrics is not None:
			state_msg += ''.join([f' {key}: {val:.2f};' for (key, val) in metrics.items()])

		pbar.set_description(state_msg)


def finetune_supervised(args, dataset, G_target):
	### create logger ###

	if not os.path.exists(f'checkpoint/{args.name}'):
		os.makedirs(f'checkpoint/{args.name}')

	if not os.path.exists(f'sample/{args.name}'):
		os.makedirs(f'sample/{args.name}')

	logger = SummaryWriter(f'checkpoint/{args.name}')

	### initialize experiment ###

	global step  # simple hack
	global batch_size  # simple hack

	step = int(math.log2(args.image_size)) - 2
	resolution = 4 * 2 ** step

	batch_size = args.batch.get(resolution, args.batch_default) * torch.cuda.device_count()  # DataParallel
	loader = sample_data(dataset, batch_size, resolution)
	loader_iter = iter(loader)

	### run experiment ###

	pbar = tqdm(range(args.phase), position=0)

	metrics = evaluate(iteration=0)
	for key, val in metrics.items():
		logger.add_scalar(key, val, 0)

	best_fid = metrics['fid']

	for i in pbar:
		adjust_lr(G_optimizer, args.lr.get(resolution, 0.001))

		### sample data and noise ###

		try:
			real_index, real_image = next(loader_iter)

		except:
			loader_iter = iter(loader)
			real_index, real_image = next(loader_iter)

		real_index = real_index.cuda()
		real_image = real_image.cuda()

		### update G (supervised) ###

		G_target.zero_grad()

		sup_loss_val = backward_G_supervised(args, G_target, real_index, real_image)

		G_optimizer.step()
		accumulate(G_running_target, G_target.module)

		### save results and checkpoints ###

		if (i + 1) % args.eval_step == 0:
			logger.add_scalar('sup_loss_val', sup_loss_val, i + 1)

			metrics = evaluate(iteration=i + 1)
			for key, val in metrics.items():
				logger.add_scalar(key, val, i + 1)

			if metrics['fid'] < best_fid:
				torch.save({
					'generator': G_target.module.state_dict(),
					'g_optimizer': G_optimizer.state_dict(),
					'g_running': G_running_target.state_dict(),
				}, f'checkpoint/{args.name}/best.model')

		state_msg = f'Size: {4 * 2 ** step}; sup: {sup_loss_val:.3f};'
		state_msg += ''.join([f' {key}: {val:.2f};' for (key, val) in metrics.items() if 'norm' not in key])  # skip norm

		pbar.set_description(state_msg)


def sample_noise(batch_size):
	if args.mixing and random.random() < 0.9:
		gen_in11, gen_in12, gen_in21, gen_in22 = torch.randn(4, batch_size, code_size, device='cuda').chunk(4, 0)
		gen_in1 = [gen_in11.squeeze(0), gen_in12.squeeze(0)]
		gen_in2 = [gen_in21.squeeze(0), gen_in22.squeeze(0)]

	else:
		gen_in1, gen_in2 = torch.randn(2, batch_size, code_size, device='cuda').chunk(2, 0)
		gen_in1 = gen_in1.squeeze(0)
		gen_in2 = gen_in2.squeeze(0)

	return gen_in1, gen_in2


def backward_D(args, G_target, D_target, real_image, gen_in):
	### update D (GAN loss) ###

	real_image = real_image.cuda()

	real_image.requires_grad = True
	real_predict = D_target(real_image, step=step, alpha=alpha)  # before activation
	D_loss_real = F.softplus(-real_predict).mean()

	grad_real = grad(outputs=real_predict.sum(), inputs=real_image, create_graph=True)[0]
	grad_penalty = (grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2).mean()
	grad_penalty = 10 / 2 * grad_penalty

	if args.miner:
		gen_in = miner(gen_in)

	fake_image_tgt = G_target(gen_in, step=step, alpha=alpha)
	fake_predict = D_target(fake_image_tgt, step=step, alpha=alpha)
	D_loss_fake = F.softplus(fake_predict).mean()

	### update D (regularizer) ###

	if args.lambda_FM > 0:
		FM_loss = FM_reg(real_image, args.feature_loc) * args.lambda_FM
	else:
		FM_loss = 0

	if args.lambda_l2_D > 0:
		l2_D_loss = l2_reg(D_source, D_target) * args.lambda_l2_D
	else:
		l2_D_loss = 0

	(D_loss_real + D_loss_fake + grad_penalty + FM_loss + l2_D_loss).backward()

	D_loss_val = (D_loss_real + D_loss_fake).item()
	grad_loss_val = grad_penalty.item() if grad_penalty > 0 else 0

	return D_loss_val, grad_loss_val


def backward_G(args, G_target, D_target, gen_in):
	### update G (GAN loss) ###

	if args.miner:
		gen_in = miner(gen_in)

	fake_image_tgt = G_target(gen_in, step=step, alpha=alpha)
	predict = D_target(fake_image_tgt, step=step, alpha=alpha)
	gen_loss = F.softplus(-predict).mean()

	### update G (regularizer) ###

	if args.lambda_l2_G > 0:
		l2_G_loss = l2_reg(G_source, G_target) * args.lambda_l2_G
	else:
		l2_G_loss = 0

	(gen_loss + l2_G_loss).backward()

	G_loss_val = gen_loss.item()

	return G_loss_val


def backward_G_supervised(args, G_target, real_index, real_image):
	### update G (supervised) ###

	real_index = real_index.cuda()
	real_image = real_image.cuda()

	embed = G_target.module.embedding(real_index)
	embed_eps = torch.randn(embed.size()).cuda() * 0.01

	if args.miner:
		gen_in = miner(embed + embed_eps)
	else:
		gen_in = embed + embed_eps

	fake_image = G_target(gen_in, step=step, alpha=alpha)

	sup_loss = criterion(fake_image, real_image, embed)

	### backward loss ###

	sup_loss.backward()

	sup_loss_val = sup_loss.item()

	return sup_loss_val


if __name__ == '__main__':
	code_size = 512
	alpha = 1  # FIX alpha = 1 (no progressive training)

	parser = argparse.ArgumentParser(description='Progressive Growing of GANs (fine-tuning)')

	parser.add_argument('--dataset', type=str, required=True, help='dataset name')
	parser.add_argument('--name', type=str, default='temp', help='name of experiment')
	parser.add_argument('--ckpt', type=str, default='./checkpoint/stylegan-256px-new.model', help='source model')
	parser.add_argument('--seed', type=int, default=0, help='random seed')

	parser.add_argument('--image_size', default=256, type=int, help='image size')
	parser.add_argument('--phase', type=int, default=50000, help='number of samples used for each training phases')
	parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
	parser.add_argument('--sched', action='store_true', help='use lr scheduling')
	parser.add_argument('--mixing', action='store_true', help='use mixing regularization')
	parser.add_argument('--loss', type=str, default='r1', choices=['r1'], help='class of gan loss')
	parser.add_argument('--eval_step', default=1000, type=int, help='step size for evaluation')
	parser.add_argument('--save_step', default=10000, type=int, help='step size for save models')
	parser.add_argument('--sample_num', default=5000, type=int, help='number of samples for evaluation')

	parser.add_argument('--init_G', action='store_true', help='initialize G')
	parser.add_argument('--init_D', action='store_true', help='initialize D')
	parser.add_argument('--only_adain', action='store_true', help='only optimize AdaIN layers')
	parser.add_argument('--supervised', action='store_true', help='use supervised loss instead of GAN loss')
	parser.add_argument('--miner', action='store_true', help='use miner network instead of full generator')
	parser.add_argument('--lambda_l2_G', type=float, default=0, help='weight for l2 loss for G')
	parser.add_argument('--lambda_l2_D', type=float, default=0, help='weight for l2 loss for D')
	parser.add_argument('--lambda_FM', type=float, default=0, help='weight for FM loss for D')
	parser.add_argument('--feature_loc', type=int, default=3, help='feature location for discriminator (default: 3)')
	parser.add_argument('--freeze_D', action='store_true', help='freeze layers of discriminator D')

	args = parser.parse_args()

	### prepare experiments ###
	random.seed(args.seed)
	torch.manual_seed(args.seed)
	torch.cuda.manual_seed_all(args.seed)

	### load dataset ###

	transform = transforms.Compose([
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
	])

	dataset = MultiResolutionDataset(f'./dataset/{args.dataset}_lmdb', transform, resolution=args.image_size)

	### load G and D ###

	if args.supervised:
		G_target = nn.DataParallel(StyledGenerator(code_size, dataset_size=len(dataset), embed_dim=code_size)).cuda()
		G_running_target = StyledGenerator(code_size, dataset_size=len(dataset), embed_dim=code_size).cuda()
		G_running_target.train(False)
		accumulate(G_running_target, G_target.module, 0)
	else:
		G_target = nn.DataParallel(StyledGenerator(code_size)).cuda()
		D_target = nn.DataParallel(Discriminator(from_rgb_activate=True)).cuda()
		G_running_target = StyledGenerator(code_size).cuda()
		G_running_target.train(False)
		accumulate(G_running_target, G_target.module, 0)

		G_source = nn.DataParallel(StyledGenerator(code_size)).cuda()
		D_source = nn.DataParallel(Discriminator(from_rgb_activate=True)).cuda()
		requires_grad(G_source, False)
		requires_grad(D_source, False)

		if args.freeze_D:
			requires_grad(D_target, False)  # freeze D

	if args.only_adain:
		for name, param in G_target.named_parameters():
			if 'adain' not in name:
				param.requires_grad = False

	if args.miner:
		requires_grad(G_target, False)
		miner = Miner(code_size).cuda()  # do not optimize G but miner
		G_optimizer = optim.Adam(miner.parameters(), lr=args.lr, betas=(0.0, 0.99))
	else:
		G_optimizer = optim.Adam(G_target.module.generator.parameters(), lr=args.lr, betas=(0.0, 0.99))
		G_optimizer.add_param_group({'params': G_target.module.style.parameters(), 'lr': args.lr * 0.01, 'mult': 0.01})

	if not args.supervised:
		D_optimizer = optim.Adam(D_target.parameters(), lr=args.lr, betas=(0.0, 0.99))

	ckpt = torch.load(args.ckpt)

	if not args.init_G:
		G_target.module.load_state_dict(ckpt['generator'], strict=False)
		G_running_target.load_state_dict(ckpt['g_running'], strict=False)

	if not args.supervised and not args.init_D:
		D_target.module.load_state_dict(ckpt['discriminator'])

	if not args.supervised:
		G_source.module.load_state_dict(ckpt['generator'])
		D_source.module.load_state_dict(ckpt['discriminator'])

	### set configs ###

	if args.sched:
		args.lr = {128: 0.0015, 256: 0.002, 512: 0.003, 1024: 0.003}
		args.batch = {4: 128, 8: 64, 16: 32, 32: 16, 64: 8, 128: 8, 256: 8}
	else:
		args.lr = {}
		args.batch = {}

	args.gen_sample = {512: (8, 4), 1024: (4, 2)}

	args.batch_default = 8

	### prepare evaluation metrics ###

	inception = nn.DataParallel(InceptionV3()).cuda()

	gen_i, gen_j = args.gen_sample.get(args.image_size, (10, 5))
	fixed_noise = torch.randn(gen_i, gen_j, code_size)

	real_images = torch.stack([dataset[i][1] for i in range(len(dataset))], dim=0)
	with open(f'./dataset/{args.dataset}_acts.pickle', 'rb') as handle:
		real_acts = pickle.load(handle)

	### run experiments ###

	if args.supervised:
		criterion = AdaBIGGANLoss(
			scale_per=0.1,
			scale_emd=0.1,
			scale_reg=0.02,
			normalize_img=1,
			normalize_per=1,
			dist_per="l2",
		).cuda()
		finetune_supervised(args, dataset, G_target)
	else:
		finetune(args, dataset, G_target, D_target)
