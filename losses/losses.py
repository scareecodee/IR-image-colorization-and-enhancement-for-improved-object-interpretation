"""
Composite Loss Functions for Thermal Colorization
"""

import torch
import torch.nn as nn
import torchvision.models as models


class VGG16FeatureExtractor(nn.Module):
    """VGG16 feature extractor for perceptual loss"""
    
    def __init__(self, layers=[4, 9, 16, 23]):
        super(VGG16FeatureExtractor, self).__init__()
        
        vgg16 = models.vgg16(pretrained=True).features
        self.slices = nn.ModuleList()
        self.means = nn.Parameter(
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            requires_grad=False
        )
        self.stds = nn.Parameter(
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            requires_grad=False
        )
        
        prev_idx = 0
        for layer_idx in layers:
            self.slices.append(
                nn.Sequential(*list(vgg16.children())[prev_idx:layer_idx+1])
            )
            prev_idx = layer_idx + 1
            
        for param in self.parameters():
            param.requires_grad = False
    
    def normalize(self, x):
        return (x - self.means) / self.stds
    
    def forward(self, x):
        x = self.normalize(x)
        features = []
        for slice_module in self.slices:
            x = slice_module(x)
            features.append(x)
        return features


class CompositeLoss(nn.Module):
    """
    Composite loss: Content + Adversarial + Perceptual + TV
    """
    
    def __init__(self, lambda_adv=0.03, lambda_perceptual=1.0, lambda_tv=1.0):
        super(CompositeLoss, self).__init__()
        
        self.lambda_adv = lambda_adv
        self.lambda_perceptual = lambda_perceptual
        self.lambda_tv = lambda_tv
        
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        self.vgg = VGG16FeatureExtractor()
    
    def content_loss(self, pred, target):
        """L1 content loss"""
        return self.l1_loss(pred, target)
    
    def adversarial_loss(self, disc_pred, target_is_real=True):
        """Adversarial loss (LSGAN)"""
        if target_is_real:
            target = torch.ones_like(disc_pred)
        else:
            target = torch.zeros_like(disc_pred)
        return self.mse_loss(disc_pred, target)
    
    def perceptual_loss(self, pred, target):
        """VGG perceptual loss"""
        pred_features = self.vgg(pred)
        target_features = self.vgg(target)
        
        loss = 0
        for p_feat, t_feat in zip(pred_features, target_features):
            loss += self.l1_loss(p_feat, t_feat)
        return loss / len(pred_features)
    
    def tv_loss(self, pred):
        """Total variation loss"""
        h_diff = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
        w_diff = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
        return (h_diff.sum() + w_diff.sum()) / (pred.shape[0] * pred.shape[1] *
                                                pred.shape[2] * pred.shape[3])
    
    def discriminator_loss(self, disc_real, disc_fake):
        """Discriminator loss"""
        real_loss = self.adversarial_loss(disc_real, True)
        fake_loss = self.adversarial_loss(disc_fake, False)
        return (real_loss + fake_loss) / 2
    
    def generator_loss(self, disc_fake, pred, target, coarse_pred=None):
        """Generator loss"""
        losses = {}
        
        losses['content'] = self.content_loss(pred, target)
        losses['adversarial'] = self.adversarial_loss(disc_fake, True) * self.lambda_adv
        losses['perceptual'] = self.perceptual_loss(pred, target) * self.lambda_perceptual
        losses['tv'] = self.tv_loss(pred) * self.lambda_tv
        
        if coarse_pred is not None:
            losses['coarse_content'] = self.content_loss(coarse_pred, target)
        
        losses['total'] = sum(losses.values())
        return losses