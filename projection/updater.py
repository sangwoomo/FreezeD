import numpy as np

import chainer
import chainer.functions as F
from chainer import Variable
from source.miscs.random_samples import sample_continuous, sample_categorical

# Classic Adversarial Loss
def loss_dcgan_dis(dis_fake, dis_real):
    L1 = F.mean(F.softplus(-dis_real))
    L2 = F.mean(F.softplus(dis_fake))
    loss = L1 + L2
    return loss


def loss_dcgan_gen(dis_fake):
    loss = F.mean(F.softplus(-dis_fake))
    return loss


# Hinge Loss
def loss_hinge_dis(dis_fake, dis_real):
    loss = F.mean(F.relu(1. - dis_real))
    loss += F.mean(F.relu(1. + dis_fake))
    return loss


def loss_hinge_gen(dis_fake):
    loss = -F.mean(dis_fake)
    return loss


class Updater(chainer.training.StandardUpdater):
    def __init__(self, *args, **kwargs):
        self.models = kwargs.pop('models')
        self.n_dis = kwargs.pop('n_dis')
        self.loss_type = kwargs.pop('loss_type')
        self.conditional = kwargs.pop('conditional')
        self.n_gen_samples = kwargs.pop('n_gen_samples')
        if self.loss_type == 'dcgan':
            self.loss_gen = loss_dcgan_gen
            self.loss_dis = loss_dcgan_dis
        elif self.loss_type == 'hinge':
            self.loss_gen = loss_hinge_gen
            self.loss_dis = loss_hinge_dis
        else:
            raise NotImplementedError
        super(Updater, self).__init__(*args, **kwargs)

    def _generete_samples(self, n_gen_samples=None):
        if n_gen_samples is None:
            n_gen_samples = self.n_gen_samples
        gen = self.models['gen']
        if self.conditional:
            y = sample_categorical(gen.n_classes, n_gen_samples, xp=gen.xp)
        else:
            y = None
        x_fake = gen(n_gen_samples, y=y)
        return x_fake, y

    def get_batch(self, xp):
        batch = self.get_iterator('main').next()
        batchsize = len(batch)
        x = []
        y = []
        for j in range(batchsize):
            x.append(np.asarray(batch[j][0]).astype("f"))
            y.append(np.asarray(batch[j][1]).astype(np.int32))
        x_real = Variable(xp.asarray(x))
        y_real = Variable(xp.asarray(y, dtype=xp.int32)) if self.conditional else None
        return x_real, y_real

    def update_core(self):
        gen = self.models['gen']
        dis = self.models['dis']
        gen_optimizer = self.get_optimizer('opt_gen')
        dis_optimizer = self.get_optimizer('opt_dis')
        xp = gen.xp
        for i in range(self.n_dis):
            if i == 0:
                x_fake, y_fake = self._generete_samples()
                dis_fake = dis(x_fake, y=y_fake)
                loss_gen = self.loss_gen(dis_fake=dis_fake)
                gen.cleargrads()
                loss_gen.backward()
                gen_optimizer.update()
                chainer.reporter.report({'loss_gen': loss_gen})

            x_real, y_real = self.get_batch(xp)
            batchsize = len(x_real)
            dis_real = dis(x_real, y=y_real)
            x_fake, y_fake = self._generete_samples(n_gen_samples=batchsize)
            dis_fake = dis(x_fake, y=y_fake)
            x_fake.unchain_backward()

            loss_dis = self.loss_dis(dis_fake=dis_fake, dis_real=dis_real)
            dis.cleargrads()
            loss_dis.backward()
            dis_optimizer.update()
            chainer.reporter.report({'loss_dis': loss_dis})


class FinetuneUpdater(chainer.training.StandardUpdater):
    def __init__(self, *args, **kwargs):
        self.models = kwargs.pop('models')
        self.n_dis = kwargs.pop('n_dis')
        self.loss_type = kwargs.pop('loss_type')
        self.conditional = kwargs.pop('conditional')
        self.n_gen_samples = kwargs.pop('n_gen_samples')
        if self.loss_type == 'dcgan':
            self.loss_gen = loss_dcgan_gen
            self.loss_dis = loss_dcgan_dis
        elif self.loss_type == 'hinge':
            self.loss_gen = loss_hinge_gen
            self.loss_dis = loss_hinge_dis
        else:
            raise NotImplementedError

        # hyper-parameters for fine-tuning
        self.FM_layer = kwargs.pop('FM_layer')
        self.lambda_FM = kwargs.pop('lambda_FM')

        super(FinetuneUpdater, self).__init__(*args, **kwargs)

    def _generete_samples(self, n_gen_samples=None):
        if n_gen_samples is None:
            n_gen_samples = self.n_gen_samples
        gen = self.models['G_tgt']
        if self.conditional:
            y = sample_categorical(gen.n_classes, n_gen_samples, xp=gen.xp)
        else:
            y = None
        x_fake = gen(n_gen_samples, y=y)
        return x_fake, y

    def get_batch(self, xp):
        batch = self.get_iterator('main').next()
        batchsize = len(batch)
        x = []
        y = []
        for j in range(batchsize):
            x.append(np.asarray(batch[j][0]).astype("f"))
            y.append(np.asarray(batch[j][1]).astype(np.int32))
        x_real = Variable(xp.asarray(x))
        y_real = Variable(xp.asarray(y, dtype=xp.int32)) if self.conditional else None
        return x_real, y_real

    def update_core(self):
        G_tgt = self.models['G_tgt']
        D_tgt = self.models['D_tgt']
        D_src = self.models['D_src']

        G_tgt_optimizer = self.get_optimizer('opt_G_tgt')
        D_tgt_optimizer = self.get_optimizer('opt_D_tgt')

        xp = G_tgt.xp
        for i in range(self.n_dis):
            if i == 0:
                # G loss
                x_fake, y_fake = self._generete_samples()
                dis_fake = D_tgt(x_fake, y=y_fake)
                loss_gen = self.loss_gen(dis_fake=dis_fake)
                G_tgt.cleargrads()
                loss_gen.backward()
                G_tgt_optimizer.update()
                chainer.reporter.report({'loss_gen': loss_gen})

            # D loss
            x_real, y_real = self.get_batch(xp)
            batchsize = len(x_real)
            dis_real = D_tgt(x_real, y=y_real)
            x_fake, y_fake = self._generete_samples(n_gen_samples=batchsize)
            dis_fake = D_tgt(x_fake, y=y_fake)
            x_fake.unchain_backward()

            loss_dis = self.loss_dis(dis_fake=dis_fake, dis_real=dis_real)
            D_tgt.cleargrads()
            loss_dis.backward()
            D_tgt_optimizer.update()
            chainer.reporter.report({'loss_dis': loss_dis})

            # FM loss
            if self.lambda_FM > 0:
                x_real, _ = self.get_batch(xp)
                feat_src = D_src(x_real, get_feature=True, layer=self.FM_layer)
                feat_tgt = D_tgt(x_real, get_feature=True, layer=self.FM_layer)
                loss_FM = F.mean_squared_error(feat_tgt, feat_src)  # MSE loss
                D_tgt.cleargrads()
                loss_FM.backward()
                D_tgt_optimizer.update()
                chainer.reporter.report({'loss_FM': loss_FM})
            else:
                chainer.reporter.report({'loss_FM': 0})

