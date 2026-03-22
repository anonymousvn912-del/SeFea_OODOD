import math

import numpy as np
import mpmath
import torch
import torch.nn as nn
import os
import torch.nn.functional as F


realmin = 1e-10


def norm(input, p=2, dim=0, eps=1e-12):
    return input.norm(p, dim, keepdim=True).clamp(min=eps).expand_as(input)

def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)
        

class vMFLogPartition(torch.autograd.Function):
    '''
    Evaluates log C_d(kappa) for vMF density
    Allows autograd wrt kappa
    '''

    besseli = np.vectorize(mpmath.besseli)
    log = np.vectorize(mpmath.log)
    nhlog2pi = -0.5 * np.log(2 * np.pi)

    @staticmethod
    def forward(ctx, *args):

        '''
        Args:
            args[0] = d; scalar (> 0)
            args[1] = kappa; (> 0) torch tensor of any shape

        Returns:
            logC = log C_d(kappa); torch tensor of the same shape as kappa
        '''

        d = args[0]
        kappa = args[1]

        s = 0.5 * d - 1

        # log I_s(kappa)
        mp_kappa = mpmath.mpf(1.0) * kappa.detach().cpu().numpy()
        mp_logI = vMFLogPartition.log(vMFLogPartition.besseli(s, mp_kappa))
        logI = torch.from_numpy(np.array(mp_logI.tolist(), dtype=float)).to(kappa)

        if (logI != logI).sum().item() > 0:  # there is nan
            raise ValueError('NaN is detected from the output of log-besseli()')

        logC = d * vMFLogPartition.nhlog2pi + s * kappa.log() - logI

        # save for backard()
        ctx.s, ctx.mp_kappa, ctx.logI = s, mp_kappa, logI

        return logC

    @staticmethod
    def backward(ctx, *grad_output):

        s, mp_kappa, logI = ctx.s, ctx.mp_kappa, ctx.logI

        # log I_{s+1}(kappa)
        mp_logI2 = vMFLogPartition.log(vMFLogPartition.besseli(s + 1, mp_kappa))
        logI2 = torch.from_numpy(np.array(mp_logI2.tolist(), dtype=float)).to(logI)

        if (logI2 != logI2).sum().item() > 0:  # there is nan
            raise ValueError('NaN is detected from the output of log-besseli()')

        dlogC_dkappa = -(logI2 - logI).exp()

        return None, grad_output[0] * dlogC_dkappa


class vMF(nn.Module):
    '''
    vMF(x; mu, kappa)
    '''

    def __init__(self, x_dim, reg=1e-6):

        super(vMF, self).__init__()

        self.x_dim = x_dim

        self.mu_unnorm = nn.Parameter(torch.randn(x_dim))
        self.logkappa = nn.Parameter(0.01 * torch.randn([]))

        self.reg = reg

    def set_params(self, mu, kappa):

        with torch.no_grad():
            self.mu_unnorm.copy_(mu)
            self.logkappa.copy_(torch.log(kappa + realmin))

    def get_params(self):

        mu = self.mu_unnorm / norm(self.mu_unnorm)
        kappa = self.logkappa.exp() + self.reg

        return mu, kappa

    def forward(self, x, utc=False):

        '''
        Evaluate logliks, log p(x)

        Args:
            x = batch for x
            utc = whether to evaluate only up to constant or exactly
                if True, no log-partition computed
                if False, exact loglik computed
        Returns:
            logliks = log p(x)
        '''

        mu, kappa = self.get_params()

        dotp = (mu.unsqueeze(0) * x).sum(1)

        if utc:
            logliks = kappa * dotp
        else:
            logC = vMFLogPartition.apply(self.x_dim, kappa)
            logliks = kappa * dotp + logC

        return logliks

    def sample(self, N=1, rsf=10):

        '''
        Args:
            N = number of samples to generate
            rsf = multiplicative factor for extra backup samples in rejection sampling

        Returns:
            samples; N samples generated

        Notes:
            no autodiff
        '''

        d = self.x_dim

        with torch.no_grad():

            mu, kappa = self.get_params()

            # Step-1: Sample uniform unit vectors in R^{d-1}
            v = torch.randn(N, d - 1).to(mu)
            v = v / norm(v, dim=1)

            # Step-2: Sample v0
            kmr = np.sqrt(4 * kappa.item() ** 2 + (d - 1) ** 2)
            bb = (kmr - 2 * kappa) / (d - 1)
            aa = (kmr + 2 * kappa + d - 1) / 4
            dd = (4 * aa * bb) / (1 + bb) - (d - 1) * np.log(d - 1)
            beta = torch.distributions.Beta(torch.tensor(0.5 * (d - 1)), torch.tensor(0.5 * (d - 1)))
            uniform = torch.distributions.Uniform(0.0, 1.0)
            v0 = torch.tensor([]).to(mu)
            while len(v0) < N:
                eps = beta.sample([1, rsf * (N - len(v0))]).squeeze().to(mu)
                uns = uniform.sample([1, rsf * (N - len(v0))]).squeeze().to(mu)
                w0 = (1 - (1 + bb) * eps) / (1 - (1 - bb) * eps)
                t0 = (2 * aa * bb) / (1 - (1 - bb) * eps)
                det = (d - 1) * t0.log() - t0 + dd - uns.log()
                copy_tensor = (w0[det >= 0]).clone().detach()
                v0 = torch.cat([v0, copy_tensor.to(mu)])
                if len(v0) > N:
                    v0 = v0[:N]
                    break
            v0 = v0.reshape([N, 1])

            # Step-3: Form x = [v0; sqrt(1-v0^2)*v]
            samples = torch.cat([v0, (1 - v0 ** 2).sqrt() * v], 1)

            # Setup-4: Householder transformation
            e1mu = torch.zeros(d, 1).to(mu);
            e1mu[0, 0] = 1.0
            e1mu = e1mu - mu if len(mu.shape) == 2 else e1mu - mu.unsqueeze(1)
            e1mu = e1mu / norm(e1mu, dim=0)
            samples = samples - 2 * (samples @ e1mu) @ e1mu.t()

        return samples


class SIREN(nn.Module):
    def __init__(self, hidden_dim, num_classes, project_dim, learnable_kappa_init=10):
        super().__init__()

        self.center_project = nn.Sequential(nn.Linear(hidden_dim, project_dim),
                                            nn.ReLU(),
                                            nn.Linear(project_dim, project_dim))

        self.learnable_kappa = nn.Linear(num_classes,1, bias=False).cuda()
        torch.nn.init.constant_(self.learnable_kappa.weight, learnable_kappa_init)
    
    def embed_features(self, osf_features):
        return self.center_project(osf_features)
    
    def forward(self, osf_features):
        output_project_features = self.center_project(osf_features)
        out = {}
        out['project_features'] = output_project_features
        out['learnable_kappa'] = self.learnable_kappa
        return out
    

class SIREN_Criterion(nn.Module):
    def __init__(self, num_classes, project_dim):
        super().__init__()
        
        self.prototypes = torch.zeros(num_classes, project_dim).cuda()
        self.project_dim = project_dim
    
    def weighted_vmf_loss(self, pred, weight_before_exp, target):
        center_adpative_weight = weight_before_exp.view(1,-1)
        pred = center_adpative_weight * pred.exp() / (
                (center_adpative_weight * pred.exp()).sum(-1)).unsqueeze(-1)
        loss  = -(pred[range(target.shape[0]), target] + 1e-6).log().mean()

        return loss

    def loss_vmf(self, outputs, target_classes_o):
        id_samples = outputs['project_features'] # nxd
        for index in range(len(target_classes_o)):
            self.prototypes.data[target_classes_o[index]] = \
                F.normalize(0.05 * F.normalize(id_samples[index], p=2, dim=-1) + \
                0.95 * self.prototypes.data[target_classes_o[index]], p=2, dim=-1)

        cosine_logits = F.cosine_similarity(self.prototypes.data.detach().unsqueeze(0).repeat(len(id_samples), 1, 1), 
                                            id_samples.unsqueeze(1).repeat(1, len(self.prototypes.data), 1), 2)

        weight_before_exp = vMFLogPartition.apply(self.project_dim, 
                                                  F.relu(outputs['learnable_kappa'].weight.view(-1, 1)))
        weight_before_exp = weight_before_exp.exp()

        cosine_similarity_loss = self.weighted_vmf_loss(
            cosine_logits * F.relu(outputs['learnable_kappa'].weight.view(1, -1)),
            weight_before_exp,
            target_classes_o)

        return cosine_similarity_loss
        

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
    