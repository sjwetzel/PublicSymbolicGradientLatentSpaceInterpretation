import torch
import numpy as np
import matplotlib.pyplot as plt

from models import SiameseNetwork
from data_utils import invariant
import os

def normalize(x, axis):
    norm = np.sqrt(np.sum(np.square(x), axis=axis, keepdims=True))
    return x/(norm + 0.00001)

def extract_latent_representation():
    # Generate data 

    x = torch.tensor(np.load('data/x_test.npy'), dtype=torch.float32).to('cuda')
    n_feats = np.shape(x)[-1]//3
    # Could also use positives/negatives, instead of anchors
    x_a, _, _ = x.chunk(3, dim=1)
    x_a.requires_grad = True

    model = SiameseNetwork(n_feats, 256, 1).to('cuda')
    # model.load_state_dict(torch.load('./model_weights/motion_energy_model.pth'))
    model.load_state_dict(torch.load('./model_weights/model.pth'))
    model.eval()




    # Obtain the latent representation of anchors, which encodes the symmetry invariant 
    model.zero_grad()
    latent_a = model.shared_dense(x_a)

    invariant_data = np.array([invariant(x) for x in x_a.cpu().detach().numpy()])
    os.makedirs('latent_data/', exist_ok=True)
    np.save('latent_data/invariant.npy', invariant_data)
    np.save('latent_data/latent.npy', latent_a.cpu().detach().numpy())
    plt.scatter(invariant_data, latent_a.cpu().detach().numpy())
    plt.show()
    
    # Obtain gradients for symbolic regression 
    latent_a.backward(torch.ones_like(latent_a))
    nn_gradients = x_a.grad.cpu().detach().numpy()
    normalized_nn_gradients = normalize(nn_gradients, axis=1)

    # Save data for symbolic regression
    np.savetxt('gradient_data/X.out', x_a.cpu().detach().numpy(), delimiter=',')
    np.savetxt('gradient_data/gradients.out', normalized_nn_gradients, delimiter=',')

if __name__ == '__main__':
    extract_latent_representation()
