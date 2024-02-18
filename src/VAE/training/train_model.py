#!/usr/bin/env python3

import math
import os

import numpy as np
import librosa as lb
import soundfile as sf
import pyaudio

import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse


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
    return (reconstruction_loss + kl_divergence)


def train(model, train_loader, epochs, device, log_file):
    '''
    trains the model
    '''
    print('Training model...', file=log_file)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()

    losses = []

    for epoch in range(epochs):
        train_loss = 0
        for batch_idx, x in enumerate(train_loader):
            x = x.to(device)
            optimizer.zero_grad()
            reconstructed_x, mu, logvar = model(x)

            loss = loss_function(reconstructed_x, x, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        average_loss = train_loss / len(train_loader.dataset)
        print('====> Epoch: {} Average loss: {:.4f}'.format(epoch+1, average_loss), file=log_file)
        losses.append(average_loss)
    print('Finished training.', file=log_file) 

    return losses

