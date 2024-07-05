import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder_fc_block(nn.Module):
    def __init__(self, 
                 height, 
                 width, 
                 channels, 
                 latent_dim, 
                 activation=nn.ReLU(), 
                 init_func= lambda x: x):
        
        super(Encoder_fc_block, self).__init__()

        self.fc_dims = width * height * channels
        self.fc = nn.Linear(self.fc_dims, latent_dim)
        self.activation = activation

        self.fc_mean = nn.Linear(latent_dim, latent_dim)
        self.fc_logvar = nn.Linear(latent_dim, latent_dim)

        for layer in [self.fc, self.fc_mean, self.fc_logvar]:
            init_func(layer.weight)
            if layer.bias is not None:
                torch.nn.init.zeros_(layer.bias)
            
        
        nn.init.normal_(self.fc_logvar.weight, mean=0.0, std=0.00001)

    def forward(self, x):
        x = x.view(-1, self.fc_dims)

        x = self.fc(x)
        x = self.activation(x)

        mu = self.fc_mean(x)
        logvar = self.fc_logvar(x)

        # print(f'x (tensor frm which logvar is made) max and min: {x.max().item()}, {x.min().item()}')

        return mu, logvar
    

class Decoder_fc_block(nn.Module):
    def __init__(self, 
                 height, 
                 width, 
                 channels, 
                 latent_dim, 
                 activation=nn.ReLU(), 
                 init_func= lambda x: x):
        super(Decoder_fc_block, self).__init__()

        self.shape = (channels, height, width)
        self.fc_dims = width * height * channels

        self.fc = nn.Linear(latent_dim, self.fc_dims)
        self.activation = activation

        for layer in [self.fc]:
            init_func(layer.weight)
            if layer.bias is not None:
                torch.nn.init.zeros_(layer.bias)

    def forward(self, x):
        x = self.fc(x)
        x = self.activation(x)

        x = x.view(-1, self.shape[0], self.shape[1], self.shape[2])

        return x



class Conv_block(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size, 
                 stride=1, 
                 padding=0, 
                 residual_padding=None, 
                 init_func= lambda x: x):
        super(Conv_block, self).__init__()

        if residual_padding is None:
            residual_padding = padding

        self.conv_1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.batch_norm_1 = nn.BatchNorm2d(out_channels)

        self.conv_2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=padding)
        self.batch_norm_2 = nn.BatchNorm2d(out_channels)

        self.conv_residual = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=residual_padding)
        self.batch_norm_residual = nn.BatchNorm2d(out_channels)

        for layer in [self.conv_1, self.conv_2, self.conv_residual]:
            init_func(layer.weight)
            if layer.bias is not None:
                torch.nn.init.zeros_(layer.bias)

        for bn in [self.batch_norm_1, self.batch_norm_2, self.batch_norm_residual]:
            torch.nn.init.ones_(bn.weight)
            torch.nn.init.zeros_(bn.bias)


    def forward(self, x):
        residual = x.clone()

        x = self.conv_1(x)
        x = self.batch_norm_1(x)
        x = F.relu(x)

        x = self.conv_2(x)
        x = self.batch_norm_2(x)

        residual = self.conv_residual(residual)
        residual = self.batch_norm_residual(residual)

        x += residual
        x = F.relu(x)

        return x


class Deconv_block(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size_1, 
                 kernel_size_2 = None, 
                 stride=1, 
                 padding_1=0, 
                 output_padding_1=0, 
                 padding_2 = None, 
                 output_padding_2= None, 
                 residual_padding=None, 
                 residual_output_padding=None,
                 init_func= lambda x: x):
        super(Deconv_block, self).__init__()

        if residual_padding is None:
            residual_padding = padding_1
        if residual_output_padding is None:
            residual_output_padding = output_padding_1

        if kernel_size_2 is None:
            kernel_size_2 = kernel_size_1
        
        if padding_2 is None:
            padding_2 = padding_1
        
        if output_padding_2 is None:
            output_padding_2 = output_padding_1


        self.deconv_1 = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size_1, stride=stride, padding=padding_1, output_padding=output_padding_1)
        self.batch_norm_1 = nn.BatchNorm2d(out_channels)

        self.deconv_2 = nn.ConvTranspose2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size_2, stride=1, padding=padding_2, output_padding=output_padding_2)
        self.batch_norm_2 = nn.BatchNorm2d(out_channels)

        self.deconv_residual = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=residual_padding, output_padding=residual_output_padding)
        self.batch_norm_residual = nn.BatchNorm2d(out_channels)

        for layer in [self.deconv_1, self.deconv_2, self.deconv_residual]:
            init_func(layer.weight)
            if layer.bias is not None:
                torch.nn.init.zeros_(layer.bias)

        for bn in [self.batch_norm_1, self.batch_norm_2, self.batch_norm_residual]:
            torch.nn.init.ones_(bn.weight)
            torch.nn.init.zeros_(bn.bias)

    def forward(self, x):
        residual = x.clone()

        x = self.deconv_1(x)
        x = self.batch_norm_1(x)
        x = F.relu(x)

        x = self.deconv_2(x)
        x = self.batch_norm_2(x)

        residual = self.deconv_residual(residual)
        residual = self.batch_norm_residual(residual)

        x += residual
        x = F.relu(x)

        return x
    

# Examples 

# Stride and no Stride deconvolution blocks that preserve the shape of the input

# deconv = Deconv_block(in_channels=10, 
#                       out_channels=1, 
#                       kernel_size_1=4,
#                       kernel_size_2=4, 
#                       stride=1, 
#                       padding_1=2, 
#                       output_padding_1=0,
#                       padding_2=1,
#                       output_padding_2=0, 
#                       residual_padding=0, 
#                       residual_output_padding=0)

# deconv_stride = Deconv_block(in_channels=10,
#                             out_channels=1,
#                             kernel_size_1=4,
#                             kernel_size_2=3,
#                             stride=2,
#                             padding_1=(2,1),
#                             output_padding_1=(1,0),
#                             padding_2=1,
#                             output_padding_2=(0,0),
#                             residual_padding=0,
#                             residual_output_padding=(0,1))

# stride and no stride convolution blocks that preserve the shape of the input

# self.block_1 = Conv_block(in_channels=1, out_channels=32, kernel_size=3, stride=2, padding=(1,1), residual_padding=0)
# self.block_2 = Conv_block(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=(1,1), residual_padding=0