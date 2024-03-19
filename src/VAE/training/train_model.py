#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F


def loss_function(reconstructed_x, x, mu, logvar):
    '''
    loss function for VAE with MSE loss as reconstruction loss and KL divergence (makes autoencoder variational)

    params:
        reconstructed_x - reconstructed x
        x - original x
        mu - mean
        logvar - log variance

    returns:
        loss - loss of the model
    '''
    reconstruction_loss = F.mse_loss(reconstructed_x, x, reduction='sum') #mse for simplicity, could change in the future
    kl_divergence = - 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    print('Reconstruction loss: ', reconstruction_loss)
    print('KL divergence: ', kl_divergence)
    print('Total loss: ', reconstruction_loss + kl_divergence)

    return (reconstruction_loss + kl_divergence)

def add_noise_to_batch(batch: torch.tensor, noise_function: callable) -> torch.tensor:
    '''
    Add noise to the batch of spectograms

    params:
        - batch: batch to add noise to
        - noise_function: function to add noise to the batch

    returns:
        - batch with added noise
    '''
    batch = batch.view(batch.shape[0], batch.shape[2], batch.shape[3])

    noised_batch = batch.clone()

    for i in range(batch.shape[0]):
        noised_batch[i,:,:] = noise_function(noised_batch[i, :, :])

    return noised_batch.view(batch.shape[0], 1, batch.shape[1], batch.shape[2])



def train(model, train_loader, epochs, device, log_file, noise_function=lambda x:x):
    '''
    trains the model

    params:
        model - model to train
        train_loader - dataloader with training data
        epochs - number of epochs to train the model
        device - device to train the model on
        log_file - file to write logs to
        noise_function - function to add noise to the spectogram

    returns:
        losses - list of losses for each epoch
    '''
    print('Training model...', file=log_file)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()

    losses = []

    for epoch in range(epochs):
        train_loss = 0
        for batch_idx, x in enumerate(train_loader):
            noised_x = noise_function(x)
            
            x = x.to(device)
            noised_x = noised_x.to(device) 

            optimizer.zero_grad()
            reconstructed_x, mu, logvar = model(noised_x)

            loss = loss_function(reconstructed_x, x, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        average_loss = train_loss / len(train_loader.dataset)
        print('====> Epoch: {} Average loss: {:.4f}'.format(epoch+1, average_loss), file=log_file)
        log_file.flush()
        
        losses.append(average_loss)
    print('Finished training.', file=log_file) 

    return losses
