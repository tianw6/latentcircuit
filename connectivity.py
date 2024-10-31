
import torch
import numpy as np
import torch
from scipy.sparse import random
from scipy import stats
from numpy import linalg

def init_connectivity( N,input_size,output_size,radius=1.5):
    '''
    Initialize connectivity of RNN
    :param N: network size
    :param input_size: number of input channels
    :param output_size: number of output channels
    :param radius: spectral radius
    :return: Connectivity and masks
    '''
    Ne = int(N * 0.8)
    Ni = int(N * 0.2)

    # Initialize W_rec
    W_rec = torch.empty([0, N])

    # Balancing parameters
    mu_E = 1 / np.sqrt(N)
    mu_I = 4 / np.sqrt(N)

    var = 1 / N

    rowE = torch.empty([Ne, 0])
    rowI = torch.empty([Ni, 0])

    rowE = torch.cat((rowE, torch.tensor(
        random(Ne, Ne, density=1, data_rvs=stats.norm(scale=var, loc=mu_E).rvs).toarray()).float()), 1)
    rowE = torch.cat((rowE, -torch.tensor(
        random(Ne, Ni, density=1, data_rvs=stats.norm(scale=var, loc=mu_I).rvs).toarray()).float()), 1)
    rowI = torch.cat((rowI, torch.tensor(
        random(Ni, Ne, density=1, data_rvs=stats.norm(scale=var, loc=mu_E).rvs).toarray()).float()), 1)
    rowI = torch.cat((rowI, -torch.tensor(
        random(Ni, Ni, density=1, data_rvs=stats.norm(scale=var, loc=mu_I).rvs).toarray()).float()), 1)

    W_rec = torch.cat((W_rec, rowE), 0)
    W_rec = torch.cat((W_rec, rowI), 0)

    W_rec = W_rec - torch.diag(torch.diag(W_rec))
    w, v = linalg.eig(W_rec)
    spec_radius = np.max(np.absolute(w))
    W_rec = radius * W_rec / spec_radius

    W_in = torch.zeros([N, input_size]).float()
    W_in[:, :] = radius * torch.tensor(
        random(N, input_size, density=1, data_rvs=stats.norm(scale=var, loc=mu_E).rvs).toarray()).float()

    W_out = torch.zeros([output_size, N])
    W_out[:, :Ne] = torch.tensor(random(output_size, Ne, density=1,
                                        data_rvs=stats.norm(scale=var, loc=mu_E).rvs).toarray()).float()

    dale_mask = torch.sign(W_rec).float()
    output_mask = (W_out != 0).float()
    input_mask = (W_in != 0).float()

    return W_rec.float(), W_in.float(), W_out.float(), dale_mask, output_mask, input_mask




