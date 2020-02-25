import torch
from torchvision import models
import torch.nn.functional as F

class Vgg16PerceptualLoss(torch.nn.Module):
    def __init__(self, perceptual_indices = [1,3,6,8,11,13,15,18,20,22] ,loss_func="l1",requires_grad = False):
        '''
        perceptual_indices: indices to use for perceptural loss
        loss_func: loss type l1 or l2
        
        Here's the list of layers and its indices. Fully connected layers are dopped.
        (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): ReLU(inplace)
        (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (3): ReLU(inplace)
        (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (6): ReLU(inplace)
        (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (8): ReLU(inplace)
        (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (11): ReLU(inplace)
        (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (13): ReLU(inplace)
        (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (15): ReLU(inplace)
        (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (18): ReLU(inplace)
        (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (20): ReLU(inplace)
        (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (22): ReLU(inplace)
        (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (25): ReLU(inplace)
        (26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (27): ReLU(inplace)
        (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (29): ReLU(inplace)
        (30): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

        '''
        super(Vgg16PerceptualLoss, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features.eval()
        max_layer_idx = max(perceptual_indices)
        self.perceptual_indices = set(perceptual_indices)#set is faster to query
        self.vgg_partial = torch.nn.Sequential(*list(vgg_pretrained_features.children())[0:max_layer_idx])
        
        if loss_func == "l1":
            self.loss_func = F.l1_loss
        elif loss_func == "l2":
            self.loss_func = F.mse_loss
        else:
            raise NotImpementedError(loss_func)
        
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def normalize(self,batch):
        '''
        normalize using imagenet mean and std
        batch: batched imagse
        '''
        mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
        return (batch - mean) / std
    
    def rescale(self,batch,lower,upper):
        '''
        rescale image to 0 to 1
        batch: batched images 
        upper: upper bound of pixel
        lower: lower bound of pixel
        '''
        return  (batch - lower)/(upper - lower)

    def forward_img(self, h):
        '''
        h: image batch
        '''        
        intermidiates = []
        for i,layer in enumerate(self.vgg_partial):
            h = layer(h)
            if i in self.perceptual_indices:
                intermidiates.append(h)
        return intermidiates    

    def forward(self, img1, img2, img1_minmax=(0,1),img2_minmax=(0,1), apply_imagenet_norm = True):
        '''
        img1: image1
        img2: image2
        img1_minmax: upper bound and lower bound of image1. default is (0,1)
        img2_minmax: upper bound and lower bound of image2. default is (0,1)
        apply_imagenet_norm: normalize using imagenet mean and std. default is True
        '''
        if img1_minmax!=(0,1):
            img1 = self.rescale(img1,img1_minmax[0],img1_minmax[1])
        if img2_minmax!=(0,1):
            img2 = self.rescale(img2,img2_minmax[0],img2_minmax[1])
            
        if apply_imagenet_norm:
            img1 = self.normalize(img1)
            img2 = self.normalize(img2)
        
        losses = []
        for img1_h,img2_h in zip(self.forward_img(img1),self.forward_img(img2)):
            losses.append(self.loss_func(img1_h,img2_h))
        
        return losses