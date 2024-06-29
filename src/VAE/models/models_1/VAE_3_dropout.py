#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F



class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()

        self.block_0 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        self.block_2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        self.block_3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        self.block_fc = nn.Sequential(
            nn.Linear(32 * 32 * 14, latent_dim),
            nn.ReLU()
        )

        self.fc_mu = nn.Linear(latent_dim, latent_dim)
        self.fc_logvar = nn.Linear(latent_dim, latent_dim)

    def forward(self, x):

        x = self.block_0(x)

        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)

        x = x.view(-1, 32 * 32 * 14)
        x = self.block_fc(x)

        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        return mu, logvar    


class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()

        self.block_fc = nn.Sequential(
            nn.Linear(latent_dim, 32 * 32 * 14),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        self.block_3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        self.block_2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        self.block_1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        self.block_0 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=5, stride=2, padding=2, output_padding=1),
        )

    def forward(self, x):
        x = self.block_fc(x)
        x = x.view(-1, 32, 32, 14)

        x = self.block_3(x)
        x = self.block_2(x)
        x = self.block_1(x)

        x = self.block_0(x)

        return x


class VAE_3_dropout(torch.nn.Module):
    input_shape = ('batch_size', 1, 256, 112)

    def __init__(self, latent_dim):
        super(VAE_3_dropout, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        reconstructed_x = self.decoder(z)
        return reconstructed_x, mu, logvar
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)
