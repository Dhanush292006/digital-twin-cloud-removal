import torch
import torch.nn.functional as F


def ssim_loss(pred, target, window=7):
    # Differentiable channelwise local SSIM approximation.
    mu_x=F.avg_pool2d(pred,window,1,window//2); mu_y=F.avg_pool2d(target,window,1,window//2)
    vx=F.avg_pool2d(pred*pred,window,1,window//2)-mu_x*mu_x
    vy=F.avg_pool2d(target*target,window,1,window//2)-mu_y*mu_y
    cov=F.avg_pool2d(pred*target,window,1,window//2)-mu_x*mu_y
    score=((2*mu_x*mu_y+0.01**2)*(2*cov+0.03**2))/((mu_x**2+mu_y**2+0.01**2)*(vx+vy+0.03**2)+1e-8)
    return 1-score.mean()


def spectral_angle_loss(pred,target):
    dot=(pred*target).sum(1); norms=pred.square().sum(1).sqrt()*target.square().sum(1).sqrt()
    return torch.acos((dot/(norms+1e-8)).clamp(-1+1e-6,1-1e-6)).mean()


def reconstruction_loss(pred,target,cloud_mask,weights=None):
    w=weights or {"global":1.,"cloud":2.,"ssim":.2,"spectral":.1}
    err=(pred-target).abs(); global_l=err.mean()
    m=(cloud_mask>0).float()[:,None]
    cloud_l=(err*m).sum()/(m.sum()*pred.shape[1]+1e-8)
    ssim=ssim_loss(pred,target); spectral=spectral_angle_loss(pred,target)
    total=w["global"]*global_l+w["cloud"]*cloud_l+w["ssim"]*ssim+w["spectral"]*spectral
    return total,{"global":global_l.detach(),"cloud":cloud_l.detach(),"ssim":ssim.detach(),"spectral":spectral.detach()}
