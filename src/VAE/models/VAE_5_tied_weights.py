#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE_5_tied_weights(nn.Module):
    '''
    input shape: (batch_size, 1, 256, 112)
    '''


    def __init__(self, latent_dim):
        super(VAE_5_tied_weights, self).__init__()
        

        self.block_0_conv = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=7, stride=2, padding=3)
        

        self.block_1_conv_1 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.block_1_conv_2 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.block_1_conv_3 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1)


        self.block_2_conv_1 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=1)
        self.block_2_conv_2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.block_2_conv_3 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)


        self.block_3_conv_1 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.block_3_conv_2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.block_3_conv_3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)


        self.block_4_conv_1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.block_4_conv_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.block_4_conv_3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)


        self.block_enc_fc = nn.Sequential(
            nn.Linear(64 * 16 * 7, latent_dim),
            nn.ReLU()
        )

        self.block_dec_fc = nn.Sequential(
            nn.Linear(latent_dim, 64 * 16 * 7),
            nn.ReLU()
        )

        self.fc_mu = nn.Linear(latent_dim, latent_dim)
        self.fc_logvar = nn.Linear(latent_dim, latent_dim)


    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed_x = self.decode(z)
        return reconstructed_x, mu, logvar
    
    
    def encode(self, x):
        x = F.relu(self.block_0_conv(x))

        x = F.relu(self.block_1_conv_1(x))
        x = F.relu(self.block_1_conv_2(x))
        x = F.relu(self.block_1_conv_3(x))

        x = F.relu(self.block_2_conv_1(x))
        x = F.relu(self.block_2_conv_2(x))
        x = F.relu(self.block_2_conv_3(x))

        x = F.relu(self.block_3_conv_1(x))
        x = F.relu(self.block_3_conv_2(x))
        x = F.relu(self.block_3_conv_3(x))

        x = F.relu(self.block_4_conv_1(x))
        x = F.relu(self.block_4_conv_2(x))
        x = F.relu(self.block_4_conv_3(x))

        x = x.view(-1, 64 * 16 * 7)

        x = self.block_enc_fc(x)

        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        return mu, logvar

        
    def decode(self, z):
        x = self.block_dec_fc(z)
        x = x.view(-1, 64, 16, 7)

        x = F.relu(F.conv_transpose2d(x, self.block_4_conv_3.weight, stride=1, padding=1))
        x = F.relu(F.conv_transpose2d(x, self.block_4_conv_2.weight, stride=1, padding=1))
        x = F.relu(F.conv_transpose2d(x, self.block_4_conv_1.weight, stride=2, padding=1, output_padding=1))

        x = F.relu(F.conv_transpose2d(x, self.block_3_conv_3.weight, stride=1, padding=1))
        x = F.relu(F.conv_transpose2d(x, self.block_3_conv_2.weight, stride=1, padding=1))
        x = F.relu(F.conv_transpose2d(x, self.block_3_conv_1.weight, stride=2, padding=1, output_padding=1))

        x = F.relu(F.conv_transpose2d(x, self.block_2_conv_3.weight, stride=1, padding=1))
        x = F.relu(F.conv_transpose2d(x, self.block_2_conv_2.weight, stride=1, padding=1))
        x = F.relu(F.conv_transpose2d(x, self.block_2_conv_1.weight, stride=2, padding=1, output_padding=1))

        x = F.relu(F.conv_transpose2d(x, self.block_1_conv_3.weight, stride=1, padding=1))
        x = F.relu(F.conv_transpose2d(x, self.block_1_conv_2.weight, stride=1, padding=1))
        x = F.relu(F.conv_transpose2d(x, self.block_1_conv_1.weight, stride=1, padding=1))

        x = F.conv_transpose2d(x, self.block_0_conv.weight, stride=2, padding=3, output_padding=1)

        return x
