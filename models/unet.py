import torch
from torch import nn


def block(a, b):
    return nn.Sequential(nn.Conv2d(a,b,3,padding=1,bias=False), nn.BatchNorm2d(b), nn.ReLU(inplace=True),
                         nn.Conv2d(b,b,3,padding=1,bias=False), nn.BatchNorm2d(b), nn.ReLU(inplace=True))


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, base=32, residual=False):
        super().__init__(); self.residual = residual
        self.e1=block(in_channels,base); self.e2=block(base,base*2); self.e3=block(base*2,base*4); self.e4=block(base*4,base*8)
        self.pool=nn.MaxPool2d(2); self.mid=block(base*8,base*16)
        self.u4=nn.ConvTranspose2d(base*16,base*8,2,2); self.d4=block(base*16,base*8)
        self.u3=nn.ConvTranspose2d(base*8,base*4,2,2); self.d3=block(base*8,base*4)
        self.u2=nn.ConvTranspose2d(base*4,base*2,2,2); self.d2=block(base*4,base*2)
        self.u1=nn.ConvTranspose2d(base*2,base,2,2); self.d1=block(base*2,base)
        self.out=nn.Conv2d(base,out_channels,1)

    def forward(self,x):
        a=self.e1(x); b=self.e2(self.pool(a)); c=self.e3(self.pool(b)); d=self.e4(self.pool(c)); z=self.mid(self.pool(d))
        z=self.d4(torch.cat([self.u4(z),d],1)); z=self.d3(torch.cat([self.u3(z),c],1))
        z=self.d2(torch.cat([self.u2(z),b],1)); z=self.d1(torch.cat([self.u1(z),a],1))
        y=self.out(z)
        return torch.clamp(y + x[:,:y.shape[1]] if self.residual else y,0,1)
