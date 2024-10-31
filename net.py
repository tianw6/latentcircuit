
import torch.nn as nn
from connectivity import *
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
#from scipy.sparse import random
from scipy import stats
from numpy import linalg



class Net(torch.nn.Module):
    # PyTorch module for implementing an RNN to be trained on cognitive tasks.
    def __init__(self, n, alpha = .2, sigma_rec=0.15, input_size=6, output_size=2,dale=False,activation = torch.nn.ReLU() ):
        super(Net, self).__init__()
        self.alpha = torch.tensor(alpha)
        self.sigma_rec = torch.tensor(sigma_rec)
        self.n = n
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        self.dale = dale
        
        # Initialize connectivity
        self.recurrent_layer = nn.Linear(self.n, self.n, bias=False)
        self.input_layer = nn.Linear(self.input_size, self.n, bias=False)
        self.output_layer = nn.Linear(self.n, self.output_size, bias=False)

        # Apply Dale's law and balance network
        if self.dale:
            self.recurrent_layer.weight.data,self.input_layer.weight.data,self.output_layer.weight.data,self.dale_mask, self.output_mask, self.input_mask = init_connectivity(self.n,self.input_size,self.output_size,device='cpu',radius=1.5)

        # Apply connectivity masks
        self.connectivity_constraints()


    # Dynamics
    def forward(self, u):

        # Get trial length
        t = u.shape[1]

        # Initialize hidden states to zero.
        states = torch.zeros(u.shape[0], 1, self.n, device=device)
        batch_size = states.shape[0]

        # Noise to be applied at each time step.
        noise = torch.sqrt(2 * self.alpha * self.sigma_rec ** 2) * torch.empty(batch_size, t, self.n).normal_(mean=0,
                                                                                                              std=1).to(
            device=device)

        # Loop over time steps
        for i in range(t - 1):
            state_new = (1 - self.alpha) * states[:, i, :] + self.alpha * (
                     self.activation(
                self.recurrent_layer(states[:, i, :]) + self.input_layer(u[:, i, :]) + noise[:, i, :]))
            states = torch.cat((states, state_new.unsqueeze_(1)), 1)
        return states

    def connectivity_constraints(self):
        # Constrain input and output to be positive
        self.input_layer.weight.data = torch.relu(self.input_layer.weight.data)
        self.output_layer.weight.data =  torch.relu(self.output_layer.weight.data)

        # Constrain network to satisfy Dale's law
        if self.dale:
            self.input_layer.weight.data = self.input_mask * torch.relu(self.input_layer.weight.data)
            self.output_layer.weight.data = self.output_mask * torch.relu(self.output_layer.weight.data)
            self.recurrent_layer.weight.data = torch.relu(
                self.recurrent_layer.weight.data * self.dale_mask) * self.dale_mask



    def l2_ortho(self):
        # Penalty to enforce orthogonality of input and output columns.
        b = torch.cat((self.input_layer.weight, self.output_layer.weight.t()), dim=1)
        b = b / torch.norm(b, dim=0)
        return torch.norm(b.t() @ b - torch.diag(torch.diag(b.t() @ b)), p=2)

    def loss_function(self, x, z, mask):
        return self.mse_z(x,z,mask) + self.l2_ortho() + 0.05 * torch.mean(x**2)

    def mse_z(self, x, z, mask):
        # Mean squared error for task performance.
        mse = nn.MSELoss()
        return mse(self.output_layer(x)*mask, z*mask)

    # Function for fitting RNN to task
    def fit(self, u, z, mask, epochs = 10000, lr=.01, verbose = False, weight_decay = 0):

        # Wrap training data as PyTorch dataset.
        my_dataset = TensorDataset(u, z,mask)
        my_dataloader = DataLoader(my_dataset, batch_size=128)

        # Initialize optimizer.
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay = weight_decay)

        # Training loop
        epoch = 0
        while epoch < epochs:
            for batch_idx, (u_batch, z_batch,mask_batch) in enumerate(my_dataloader):
                optimizer.zero_grad()
                x_batch = self.forward(u_batch)
                loss = self.loss_function(x_batch, z_batch, mask_batch)
                loss.backward()
                optimizer.step()

                # Apply connectivity constraints after each gradient step.
                self.connectivity_constraints()

            epoch += 1
            if verbose:
                if epoch % 5 == 0:
                    x = self.forward(u)
                    print('Epoch: {}/{}.............'.format(epoch, epochs), end=' ')
                    print("mse_z: {:.4f}".format(self.mse_z(x, z, mask).item()))
                

