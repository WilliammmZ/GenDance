import torch
import torch.nn as nn
import copy
import math
from torch.functional import F
from torch.autograd import Variable

#initial pose VAE

class InitPose_Enc(nn.Module):
    def __init__(self, pose_size, dim_z_init):
        super(InitPose_Enc, self).__init__()
        nf = 64
        self.enc = nn.Sequential(
          nn.Linear(pose_size, nf),
          nn.LayerNorm(nf),
          nn.LeakyReLU(0.2, inplace=True),
          nn.Linear(nf, nf),
          nn.LayerNorm(nf),
          nn.LeakyReLU(0.2, inplace=True),
        )
        self.mean = nn.Sequential(
          nn.Linear(nf,dim_z_init),
        )
        self.std = nn.Sequential(
          nn.Linear(nf,dim_z_init),
        )
    def forward(self, pose):
        enc = self.enc(pose)
        return self.mean(enc), self.std(enc)

class InitPose_Dec(nn.Module):
    def __init__(self, pose_size, dim_z_init):
        super(InitPose_Dec, self).__init__()
        nf = 64
        #nf = dim_z_init
        self.dec = nn.Sequential(
          nn.Linear(dim_z_init, nf),
          nn.LayerNorm(nf),
          nn.LeakyReLU(0.2, inplace=True),
          nn.Linear(nf, nf),
          nn.LayerNorm(nf),
          nn.LeakyReLU(0.2, inplace=True),
          nn.Linear(nf,pose_size),
        )
    def forward(self, z_init):
        return self.dec(z_init)

class PoseVae(nn.Module):
    def __init__(self, pose_size, dim_z_init):
        super(PoseVae, self).__init__()
        self.encoder = InitPose_Enc(pose_size, dim_z_init)
        self.decoder = InitPose_Dec(pose_size, dim_z_init)
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var/2)
        eps = torch.randn_like(std)
        
        return mu + eps * std
    
    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decoder(z)

        return x_recon, mu, log_var
    
    def decode(self, x):
        x_recon = self.decoder(x)
        return x_recon
    def encode(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        return z
        
