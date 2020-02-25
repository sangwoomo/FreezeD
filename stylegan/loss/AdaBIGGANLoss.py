import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from .Vgg16PerceptualLoss import Vgg16PerceptualLoss

class AdaBIGGANLoss(nn.Module):
    def __init__(self,perceptual_loss = "vgg",
                 scale_per=0.001,
                 scale_emd=0.1,
                 scale_reg=0.02,
                 normalize_img = True,
                 normalize_per = False,
                 dist_per = "l1",
                ):
        '''
        perceptual_loss: preceptural loss
        perceptual_facter: 
        '''
        super(AdaBIGGANLoss,self).__init__()
        if perceptual_loss == "vgg":
            self.perceptual_loss =  Vgg16PerceptualLoss(loss_func=dist_per)
        else:
            self.perceptual_loss =  perceptual_loss
        self.scale_per = scale_per
        self.scale_emd = scale_emd
        self.scale_reg = scale_reg
        self.normalize_img = normalize_img
        self.normalize_perceptural = normalize_per
        
    def earth_mover_dist(self,z):
        """
        taken from https://github.com/nogu-atsu/SmallGAN/blob/f604cd17516963d8eec292f3faddd70c227b609a/gen_models/ada_generator.py#L150-L162
        earth mover distance between z and standard normal distribution
        """
        dim_z = z.shape[1]
        n = z.shape[0]#batchsize
        t = torch.randn((n * 10,dim_z),device=z.device)
        dot = torch.matmul(z, t.permute(-1, -2))
        
        #in the original implementation transb=True
        #so we want to do t = t.swapaxes(-1, -2)
        #from https://github.com/chainer/chainer/blob/c2cf7fb9c49cf98a94caf453f644d612ace45625/chainer/functions/math/matmul.py#L
        #then swapaxes is .permute 
        #from https://discuss.pytorch.org/t/swap-axes-in-pytorch/970
        
        dist = torch.sum(z ** 2, dim=1, keepdim=True) - 2 * dot + torch.sum(t ** 2, dim=1)
        
        return torch.mean(dist.min(dim=0)[0]) + torch.mean(dist.min(dim=1)[0])

    def l1_reg(self,W):
        #https://github.com/nogu-atsu/SmallGAN/blob/2293700dce1e2cd97e25148543532814659516bd/gen_models/ada_generator.py#L146-L148
        #NOTE: I think this should be implemented as weight decay in the optimizer. It's not beatiful code to pass W into loss function.
        return torch.mean( W ** 2 )

    def forward(self,x,y,z):
        #from IPython import embed;embed()
        '''
        x:generated image. shape is (batch,channel,h,w)
        y:target image. shape is (batch,channel,h,w)
        z: seed image embeddings (BEFORE adding the noise of eps). shape is (batch,embedding_dim)
        W: model.linear.weight -> StyleGAN is unconditional! (no class embedding)
        see the equation (3) in the paper
        '''
        
        # F.mse_loss is L2 loss
        # F.l1_loss is L1 loss       

        # print(f'{x.max()}\t{x.min()}\t{y.max()}\t{x.min()}')

        #pytorch regards an image as a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
        #(see transforms.ToTensor() for details)
        #but the model output uses tanh  so x is ranging (-1 to 1) 
        #so let's rescale y to (-1 to 1) from (0 to 1)
        #chainer implementation use (-1,1) for loss computation, so i didn't do the other way around (i.e. scale x to (0,1))
        image_loss = F.l1_loss(x, y)
        if self.normalize_img:
            loss = image_loss/image_loss.item()
        else:
            loss = image_loss
        #rescaled to 1 in the chainer code
        #see https://github.com/nogu-atsu/SmallGAN/blob/2293700dce1e2cd97e25148543532814659516bd/gen_models/ada_generator.py#L68-L69
        
        for ploss in self.perceptual_loss(img1=x,img2=y,img1_minmax=(-1,1),img2_minmax=(-1,1)):
            if self.normalize_perceptural:
                loss += self.scale_per*ploss/ploss.item()
            else:
                loss += self.scale_per*ploss
            
        loss += self.scale_emd*self.earth_mover_dist(z)
        
        return  loss