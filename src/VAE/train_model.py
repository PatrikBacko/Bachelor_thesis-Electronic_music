#!/usr/bin/env python3
'''
Script for training the model (Variational Autoencoder) on given samples and saving it.
'''

import math
import os

import numpy as np
import librosa as lb
import soundfile as sf
import pyaudio

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.VAE.VAE_1 import VAE_1

import argparse

def build_arguments():
    '''
    builds arguments for the script
    '''

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('source_dir', type=str, help='Path to directory with samples.')
    parser.add_argument('output_path', type=str, help='Path to directory where to save the model and logs etc.')

    parser.add_argument('--model_name', type=str, help='Name of the log file.')
    parser.add_argument('--sample_group', type=str, help='Name of the sample group to train on.')

    parser.add_argument('--model', type=str, default='VAE_1',help='Model to train.')
    parser.add_argument('--latent_dim', type=int, default=32, help='Latent dimension of the model.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train the model.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training the model.')

    return parser.parse_args()



def pad_or_trim(mfcc, length = 100):
    '''
    pads or trims mfcc to given length, default is 100 ! (cca 1 second with 256 hop length and 512 n_fft and 44100 sr) !

    params:
        mfcc - mfcc to pad or trim
        length - length to pad or trim to (default is 100)

    returns:
        mfcc - padded or trimmed mfcc
    '''

    if mfcc.shape[1] > length:
        return mfcc[:, :length]
    else:
        last_column = mfcc[:, -1:]
        padding = np.repeat(last_column, length - mfcc.shape[1], axis=1)
        return np.concatenate((mfcc, padding), axis=1)

def prepare_data(source_dir, log_file, batch_size = 32, length = 100):
    '''
    gets path to directory with samples, and returns a dataloader with padded or trimmed mfccs of the samples
    ! padded or trimmed to cca 1 second with 256 hop length and 512 n_fft and 44100 sr !

    params:
        source_dir - path to directory with samples
        length - length to pad or trim to (default is 100)

    returns:
        train_loader - dataloader with padded or trimmed mfccs of the samples
    '''
    print(f'Preparing data from {source_dir}...', file=log_file)

    paths_to_samples = [os.path.join(source_dir, path) for path in os.listdir(source_dir)]

    mfccs = []

    for path in paths_to_samples:
            array, sr = lb.load(path)

            mfcc = lb.feature.mfcc(y=array, sr=sr, n_mfcc=512, n_fft=512, hop_length=256, lifter=0, dct_type=3, n_mels = 256)
            mfcc_pad_or_trim = pad_or_trim(mfcc, 100)

            mfccs.append(mfcc_pad_or_trim)

    mfccs_tensor = torch.tensor(np.array(mfccs)).view(-1, 1, 256, 100)
    train_loader = torch.utils.data.DataLoader(mfccs_tensor, batch_size=batch_size, shuffle=True)

    print('Data prepared.', file=log_file)

    return train_loader


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

def log_model_config(args, config_file):
    '''
    logs config of the model
    '''
    print('******************', file=config_file)
    print(f'Model {args.model_name} config:', file=config_file)
    print('******************', file=config_file)
    print('Model: ', args.model, file=config_file)
    print('Latent dimension: ', args.latent_dim, file=config_file)
    print('Epochs: ', args.epochs, file=config_file)
    print('Batch size: ', args.batch_size, files=config_file)
    print('******************', file=config_file)
    
# TODO:implement function for loading data from all sample groups at once
def main(args):
    data_path = os.path.join(args.source_dir, args.sample_group)
    log_file = open(os.path.join(args.output_path, f'{args.model_name}_training.log'), 'a')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    train_loader = prepare_data(data_path, args.batch_size, length = 100)
    ##TODO: check if this is correct
    model = VAE_1(args.latent_dim).to(device)

    ### debug log ###
    print('$$$$$$$$$$$$$$$$$$$$$$$', file=log_file)
    print('Device: ', device, file=log_file)
    print('$$$$$$$$$$$$$$$$$$$$$$$', file=log_file)
    #################

    losses = train(model, train_loader, args.epochs, device)

    torch.save(model.state_dict(), args.output_path)
    print(f'Model saved to {args.output_path}', file=log_file)

    log_file.close()

    with open(os.path.join(args.output_path, f'{args.model_name}_config'), 'a') as config_file:
        log_model_config(args, config_file)


if __name__ == '__main__':
    args = build_arguments()
    main(args)
