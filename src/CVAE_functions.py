import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader 

import matplotlib.pyplot as plt
import numpy as np


class SignalDataset_v4(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = self.data[idx][0] #load pair x0/x1
        y = self.data[idx][1]
        return x, y  

# define the loss function
def loss_function(recon_x, x, cond_data, mu, logvar, beta, wx, wy):
    
    recon_loss_fn = torch.nn.L1Loss(reduction='mean')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    x_loss  = recon_loss_fn(x, recon_x)
    recon_cond_data = recon_x[:,0,:,0]*recon_x[:,0,:,1]
    y_loss =  recon_loss_fn(cond_data, recon_cond_data)
    total_loss = (beta * KLD + wx * x_loss + wy * y_loss).mean()
    return total_loss#, x_loss, y_loss


def train_cvae(cvae, train_loader, optimizer, beta, wx, wy, epoch, device):
    cvae.train()
    train_loss = 0.0
    recon_loss = 0.0
    cond_loss = 0.0

    for batch_idx, (data, cond_data) in enumerate(train_loader):
        
        data = data.to(device)
        cond_data = cond_data.to(device)
        # ===================forward=====================
        recon_data,  z_mean, z_logvar = cvae(data, cond_data)
        loss = loss_function(recon_data, data, cond_data, z_mean, z_logvar, beta, wx, wy)
        train_loss += loss.item()
        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= len(train_loader)
    print('Train Epoch {}: Average Loss: {:.6f}'.format(epoch, train_loss))
    
    

def test_cvae(cvae, test_loader, beta, wx, wy, device):
    cvae.eval()
    test_loss = 0.0

    with torch.no_grad():
        
        for batch_idx, (data, cond_data) in enumerate(test_loader):
            data = data.to(device)
            cond_data = cond_data.to(device)

            recon_data, z_mean, z_logvar = cvae(data, cond_data)

            loss = loss_function(recon_data, data, cond_data, z_mean, z_logvar, beta, wx, wy)

            test_loss += loss.item()

    test_loss /= len(test_loader)
    
    print('Test Loss: {:.6f}'.format(test_loss))
    
    
def generate_samples(cvae, num_samples, given_y, input_shape, device, zmult = 1):
    
    cvae.eval()
    samples = []
    givens = []
    
    with torch.no_grad():
        for _ in range(num_samples):
            # Generate random latent vector
            
            z_rand = (torch.randn(*input_shape)*zmult).to(device)
            num_args = cvae.encoder.forward.__code__.co_argcount
            if num_args > 2 :
                z = cvae.sampling(*cvae.encoder(z_rand.unsqueeze(0), given_y))
            else: 
                z = cvae.sampling(*cvae.encoder(z_rand.unsqueeze(0)))
            # Another way to generate random latent vector
            #z = torch.randn(1, latent_dim).cuda()
            
            # Set conditional data as one of the given y 
            # Generate sample from decoder under given_y
            sample = cvae.decoder(z, given_y)
            samples.append(sample)
            givens.append(given_y)

    
    samples = torch.cat(samples, dim=0)   
    givens = torch.cat(givens, dim=0) 
    return samples, givens
    

    
def plot_samples(x, y, num_samples , n_cols = 10, fig_size = 2): 
    
    x = x[0:num_samples]
    y = y[0:num_samples]
    n_rows = round(len(x)/n_cols)
    
    plt.rcdefaults()
    f, axarr = plt.subplots(n_rows, n_cols, figsize=(1.25*n_cols*fig_size, n_rows*fig_size))

     
    for j, ax in enumerate(axarr.flat):
        x0 = x[j,0,:,0].cpu().detach().numpy().flatten()
        x1 = x[j,0,:,1].cpu().detach().numpy().flatten()
        given_y = y[j].cpu().detach().numpy().flatten()
        
        y_gen = x0*x1
        
        ax.plot(range(50),x0)
        ax.plot(range(50),x1)
        ax.plot(range(50),y_gen)
        ax.plot(range(50),given_y, color = 'r', linestyle = 'dotted')  
        
        ax.set_xticks([])
        ax.set_yticks([])
        

    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    plt.show() 
    
def plot_samples_stacked(x, y, n_cols = 3, fig_size = 3): 
   
    plt.rcdefaults()
    f, axs = plt.subplots(1, n_cols, figsize=(1.25*n_cols*fig_size, fig_size))
   
    
    for j in range(len(x)):
        
        x0 = x[j,0,:,0].cpu().numpy().flatten()
        x1 = x[j,0,:,1].cpu().numpy().flatten()
        y_given = y[j].cpu().numpy().flatten()
        y_gen = x0*x1
  
        axs[0].plot(range(50),x0)
        axs[0].set_ylim(0,1) 
        axs[0].set(xticks=[], yticks=[])
        axs[0].set_title('X0') 
            
        axs[1].plot(range(50),x1)
        axs[1].set_ylim(0,1) 
        axs[1].set(xticks=[], yticks=[])
        axs[1].set_title('X1') 

        axs[2].plot(range(50),y_given,color = 'r')
        axs[2].plot(range(50),y_gen, color = 'g', linestyle = 'dotted')
        axs[2].set_ylim(0,1) 
        axs[2].set(xticks=[], yticks=[])
        axs[2].set_title('Y (given and recon)')
        axs[2].legend(['y_given', 'y_recon'])
    
    plt.show()   
    