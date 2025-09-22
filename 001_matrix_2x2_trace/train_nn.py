
from models import SiameseNetwork
from data_utils import invariant
import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

import matplotlib.pyplot as plt
import os


def latent_projection(iter, loss):

        

    # Set random seed for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    np.random.seed(42)

    # Generate data 
    x = torch.tensor(np.load('data/x_test.npy'), dtype=torch.float32).to('cuda')

    # Could also use positives/negatives, instead of anchors
    x_a, _, _ = x.chunk(3, dim=1)
    x_a.requires_grad = True

    # load model
    n_feats = x_a.shape[-1]

    model = SiameseNetwork(n_feats, 256, 1).to('cuda')
    model.load_state_dict(torch.load(f'./model_run/models/model_{iter}.pth'))
    model.eval()



    # Obtain the latent representation of anchors, which encodes the symmetry invariant 
    model.zero_grad()
    latent_a = model.shared_dense(x_a)

    # Calculate spacetime interval and plot - should have a linear relationship
    E = np.array([invariant(x) for x in x_a.cpu().detach().numpy()])
    
    plt.scatter(E, latent_a.cpu().detach().numpy())
    plt.title(f'Val Loss: {str(loss)}')
    plt.xlabel('Invariant')
    plt.ylabel('Latent Representation')
    plt.savefig(f'model_run/images/plot_{iter}.png')
    plt.close()


def train():

    # Set random seed for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    np.random.seed(42)

    x_train = np.load('data/x_train.npy')
    x_val = np.load('data/x_val.npy')
    
   
    # extract num features from triplets
    n_feats = x_train.shape[-1]//3
    l2 = 0.0001
    lr = 0.001

    # Define model, loss and optimizer
    model = SiameseNetwork(n_feats, 256, 1).to('cuda')
    criterion = torch.nn.TripletMarginLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.2, patience=10)

    batch_size = 256
    epochs = 100

    train_losses = []
    val_losses = []

    for epoch in range(epochs):

        # Training
        model.train()
        train_loss = 0 
        
        idxs = np.arange(len(x_train))
        np.random.shuffle(idxs)
        x_train = x_train[idxs]

        for i in range(0, len(x_train), batch_size):

            x_train_batch = torch.tensor(x_train[i:i+batch_size], dtype=torch.float32).to('cuda')

            optimizer.zero_grad()

            latent_a, latent_p, latent_n = model(x_train_batch)
            loss = criterion(latent_a, latent_p, latent_n)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_size

        train_loss = train_loss/len(x_train)
        train_losses.append(train_loss)

        if epoch % 5 == 0:
            print(f'Epoch {epoch} training loss: {train_loss}')

        # Validation 
        model.eval()
        val_loss = 0

        for i in range(0, len(x_val), batch_size):

            x_val_batch = torch.tensor(x_val[i:i+batch_size], dtype=torch.float32).to('cuda')

            latent_a, latent_p, latent_n = model(x_val_batch)
            loss = criterion(latent_a, latent_p, latent_n)
            
            val_loss += loss.item() * batch_size

        val_loss = val_loss/len(x_val)
        val_losses.append(val_loss)
        
        if epoch % 5 == 0:
            os.makedirs('model_run/models', exist_ok=True)
            os.makedirs('model_run/images', exist_ok=True)
            torch.save(model.state_dict(), f'model_run/models/model_{epoch}.pth')
            print(f'Epoch {epoch} validation loss: {val_loss}')
            latent_projection( epoch, val_loss)

        scheduler.step(val_loss)
        #print('Last LR:', scheduler.get_last_lr())

    # Save model
    os.makedirs('model_weights/', exist_ok=True)
    torch.save(model.state_dict(), 'model_weights/model.pth')
    
    plt.plot(train_losses, label='train')
    plt.plot(val_losses, label='val')
    plt.legend()

    # Save hyperparameters
    hyperparameters = {
        'l2': l2,
        'lr': lr,
        'batch_size': batch_size,
        'epochs': epochs,
    }

    # Create directory if it doesn't exist
    os.makedirs('runs/', exist_ok=True)

    with open('runs/hyperparameters.txt', 'w') as f:
        for key, value in hyperparameters.items():
            f.write(f'{key}: {value}\n')

    # Save loss curve
    plt.savefig('runs/loss_curve.png')


if __name__ == '__main__':
    train()