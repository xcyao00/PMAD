import math
import torch
import torch.nn as nn
import FrEIA.framework as Ff
import FrEIA.modules as Fm


def get_logp(C, z, logdet_J):
    _GCONST_ = -0.9189385332046727 # ln(sqrt(2*pi))
    logp = C * _GCONST_ - 0.5*torch.sum(z**2, 1) + logdet_J
    return logp

log_theta = nn.LogSigmoid()


def subnet_fc(dims_in, dims_out):
    return nn.Sequential(nn.Linear(dims_in, 2*dims_in), nn.ReLU(), nn.Linear(2*dims_in, dims_out))

def freia_flow_head(c, n_feat):
    coder = Ff.SequenceINN(n_feat)
    print('NormFlow Coder: feature dim: ', n_feat)
    for k in range(c.coupling_blocks):
        coder.append(Fm.AllInOneBlock, subnet_constructor=subnet_fc, affine_clamping=c.clamp_alpha,
            global_affine_type='SOFTPLUS', permute_soft=True)
    return coder


def freia_cflow_head(c, n_feat, dim_c):
    coder = Ff.SequenceINN(n_feat)
    print('Condition NormFlow Coder: feature dim: ', n_feat)
    for k in range(c.coupling_blocks):  # 8
        coder.append(Fm.AllInOneBlock, cond=0, cond_shape=(dim_c,), subnet_constructor=subnet_fc, affine_clamping=c.clamp_alpha,
            global_affine_type='SOFTPLUS', permute_soft=True)
    return coder


def load_decoder_arch(c, dim_in, dim_c=128):
    if c.dec_arch == 'freia-flow':
        decoder = freia_flow_head(c, dim_in)
    elif c.dec_arch == 'freia-cflow':
        decoder = freia_cflow_head(c, dim_in, dim_c)
    else:
        raise NotImplementedError('{} is not supported NF!'.format(c.dec_arch))
    
    return decoder


def positionalencoding2d(D, H, W):
    """
    :param D: dimension of the model
    :param H: H of the positions
    :param W: W of the positions
    :return: DxHxW position matrix
    """
    if D % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with odd dimension (got dim={:d})".format(D))
    P = torch.zeros(D, H, W)
    # Each dimension use half of D
    D = D // 2
    div_term = torch.exp(torch.arange(0.0, D, 2) * -(math.log(1e4) / D))
    pos_w = torch.arange(0.0, W).unsqueeze(1)
    pos_h = torch.arange(0.0, H).unsqueeze(1)
    P[0:D:2, :, :]  = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, H, 1)
    P[1:D:2, :, :]  = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, H, 1)
    P[D::2,  :, :]  = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, W)
    P[D+1::2,:, :]  = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, W)
    return P