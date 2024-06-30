#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.VAE.models.models_magpie.blocks import Conv_block, Deconv_block, Encoder_fc_block, Decoder_fc_block


class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()

        init = lambda x: nn.init.kaiming_normal_(x, mode='fan_out', nonlinearity='relu')
        # init = lambda x: x

        self.block_1 = Conv_block(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=(1,1), residual_padding=0, init_func=init)

        self.block_2 = Conv_block(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=(1,1), residual_padding=0, init_func=init)
        # self.block_2 = Conv_block(in_channels=1, out_channels=32, kernel_size=3, stride=2, padding=(1,1), residual_padding=0, init_func=init)
        self.block_3 = Conv_block(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=(1,1), residual_padding=0, init_func=init)

        self.block_4 = Conv_block(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=(1,1), residual_padding=0, init_func=init)
        self.block_5 = Conv_block(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=(1,1), residual_padding=0, init_func=init)

        self.block_6 = Conv_block(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=(1,1), residual_padding=0, init_func=init)
        self.block_7 = Conv_block(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=(1,1), residual_padding=0, init_func=init)

        self.block_8 = Conv_block(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=(1,1), residual_padding=0, init_func=init)
        self.block_9 = Conv_block(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=(1,1), residual_padding=0, init_func=init)

        # self.block_fc = Encoder_fc_block(height=17, width=16, channels=256, latent_dim=latent_dim, init_func=init)
        self.block_fc = Encoder_fc_block(height=17, width=16, channels=256, latent_dim=latent_dim)


    def forward(self, x):
        x = self.block_1(x)

        x = self.block_2(x)
        x = self.block_3(x)

        x = self.block_4(x)
        x = self.block_5(x)

        x = self.block_6(x)
        x = self.block_7(x)

        x = self.block_8(x)
        x = self.block_9(x)

        mu, logvar = self.block_fc(x)

        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()

        init = lambda x: nn.init.kaiming_normal_(x, mode='fan_out', nonlinearity='relu')
        # init = lambda x: x

        self.block_fc = Decoder_fc_block(height=17, width=16, channels=256, latent_dim=latent_dim)

        self.block_1 = Deconv_block(in_channels=256, 
                                    out_channels=256, 
                                    kernel_size_1=4,
                                    kernel_size_2=4, 
                                    stride=1, 
                                    padding_1=2, 
                                    output_padding_1=0,
                                    padding_2=1,
                                    output_padding_2=0, 
                                    residual_padding=0, 
                                    residual_output_padding=0)

        self.block_2 = Deconv_block(in_channels=256,
                                    out_channels=128,
                                    kernel_size_1=4,
                                    kernel_size_2=3,
                                    stride=2,
                                    padding_1=(2,1),
                                    output_padding_1=(1,0),
                                    padding_2=1,
                                    output_padding_2=(0,0),
                                    residual_padding=0,
                                    residual_output_padding=(0,1))
        
        self.block_3 = Deconv_block(in_channels=128,
                                    out_channels=128,
                                    kernel_size_1=4,
                                    kernel_size_2=4,
                                    stride=1,
                                    padding_1=2,
                                    output_padding_1=0,
                                    padding_2=1,
                                    output_padding_2=0,
                                    residual_padding=0,
                                    residual_output_padding=0)
        
        self.block_4 = Deconv_block(in_channels=128,
                                    out_channels=64,
                                    kernel_size_1=4,
                                    kernel_size_2=3,
                                    stride=2,
                                    padding_1=(2,1),
                                    output_padding_1=(1,0),
                                    padding_2=1,
                                    output_padding_2=(0,0),
                                    residual_padding=0,
                                    residual_output_padding=(0,1))
        
        self.block_5 = Deconv_block(in_channels=64,
                                    out_channels=64,
                                    kernel_size_1=4,
                                    kernel_size_2=4,
                                    stride=1,
                                    padding_1=2,
                                    output_padding_1=0,
                                    padding_2=1,
                                    output_padding_2=0,
                                    residual_padding=0,
                                    residual_output_padding=0)
        
        self.block_6 = Deconv_block(in_channels=64,
                                    out_channels=32,
                                    kernel_size_1=4,
                                    kernel_size_2=3,
                                    stride=2,
                                    padding_1=(2,1),
                                    output_padding_1=(1,0),
                                    padding_2=1,
                                    output_padding_2=(0,0),
                                    residual_padding=0,
                                    residual_output_padding=(0,1))
        
        self.block_7 = Deconv_block(in_channels=32,
                                    out_channels=32,
                                    kernel_size_1=4,
                                    kernel_size_2=4,
                                    stride=1,
                                    padding_1=2,
                                    output_padding_1=0,
                                    padding_2=1,
                                    output_padding_2=0,
                                    residual_padding=0,
                                    residual_output_padding=0)
        
        self.block_8 = Deconv_block(in_channels=32,
                                    out_channels=16,
                                    kernel_size_1=4,
                                    kernel_size_2=3,
                                    stride=2,
                                    padding_1=(2,1),
                                    output_padding_1=(1,0),
                                    padding_2=1,
                                    output_padding_2=(0,0),
                                    residual_padding=0,
                                    residual_output_padding=(0,1))
        
        self.block_9 = Deconv_block(in_channels=16,
                                    out_channels=16,
                                    kernel_size_1=4,
                                    kernel_size_2=4,
                                    stride=1,
                                    padding_1=2,
                                    output_padding_1=0,
                                    padding_2=1,
                                    output_padding_2=0,
                                    residual_padding=0,
                                    residual_output_padding=0)
        
        self.final_conv = nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=3, stride=1, padding=1, output_padding=0)

    def forward(self, x):
        x = self.block_fc(x)

        x = self.block_1(x)
        x = self.block_2(x)

        x = self.block_3(x)
        x = self.block_4(x)

        x = self.block_5(x)
        x = self.block_6(x)

        x = self.block_7(x)
        x = self.block_8(x)

        x = self.block_9(x)
        x = self.final_conv(x)

        return x



class VAE_magpie_1(nn.Module):
    input_shape = ('batch_size', 1, 257, 256)

    def __init__(self, latent_dim):
        super(VAE_magpie_1, self).__init__()
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
        print()
        print(f'logvar max and min: {logvar.max().item()}, {logvar.min().item()}')
        print(f'mu max and min: {mu.max().item()}, {mu.min().item()}')
        print(f'z max and min: {z.max().item()}, {z.min().item()}')
        print(f'reconstructed_x max and min: {reconstructed_x.max().item()}, {reconstructed_x.min().item()}')
        print()
        return reconstructed_x, mu, logvar
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)
