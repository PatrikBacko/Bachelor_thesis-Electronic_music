 #!/usr/bin/env python3
'''
Script for training the model (Variational Autoencoder) on given samples and saving it.
usage: python main.py [data_dir] [output_path] [--model_name]  [-h] [optional args] 
'''



import sys
# sys.path.append(r'C:\Users\llama\Desktop\cuni\bakalarka\Bachelor_thesis-Electronic_music')


import os
import torch
import argparse
import datetime
import matplotlib.pyplot as plt
from typing import Sequence

from src.VAE.training.train_model import train
from src.VAE.utils.data import MFCC_KWARGS
from src.VAE.training.prepare_data_train import prepare_train_loader
from src.VAE.utils.add_noise import generate_noise, NOISE_SCOPE, NOISE_OPERATION_TYPES, NOISE_GENERATING_DISTS
from src.VAE.utils.config import save_config
from src.VAE.models.load_model import MODELS

from src.VAE.models.load_model import create_model
from src.VAE.utils.scaler import create_scaler



def parse_arguments():
    '''
    builds arguments for the script
    '''

    parser = argparse.ArgumentParser(description=__doc__)

    #required arguments
    parser.add_argument('data_dir', type=str, help='Required, Path to directory with samples.')
    parser.add_argument('output_path', type=str, help='Required, Path to directory where to save the model and logs etc.')
    parser.add_argument('--model_name', type=str, help='Required, Name of the log file.')
    parser.add_argument('--model', type=str, choices= MODELS, help='Model to train.')

    #optional arguments
    parser.add_argument('--sample_group', type=str, default='all', help='Names of the sample groups to train the model on, seperated by comma.'
                                                            'If used, only the specified sample groups will be used for training. (default is all)'
                                                            'possible groups: kick, clap, hat, snare, tom, cymbal, crash, ride')
    parser.add_argument('--latent_dim', type=int, default=32, help='Latent dimension of the model. (default is 32)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train the model. (default is 100)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training the model. (default is 32)')
    parser.add_argument('--pad_or_trim_length', type=int, default=112, help='Length of the spectogram to be trimmed or padded to. '
                                                                '(default is 112, With the current settings of mfcc conversion, it is around 1 seconds of audio, and it is divisible by 2 several times, which is useful for the model.)')

    parser.add_argument("--kl_regularisation", help="KL divergence regularisation. (default is 1.0)", type=float, default=1.0)
    parser.add_argument("--learning_rate", help="Learning rate for the model. (default is 0.001)", type=float, default=0.001)

    parser.add_argument("--scaler", help="scaler for the data. (default is None)", type=str, default='None', choices=['None', 'standard'])
    
    
    #Noise arguments
    parser.add_argument('-n','--noise', type=bool, default=False, help='Add noise to the spectograms. Config of noise can be set with other noise arguments.'
                        'if this argument is False, other arguments will be ignored. (default is False)\n')
    
    #argument for noise variance
    parser.add_argument("-v", "--variance", help="noise variance for generating distribution. (default is 0)", type=float, default=0.0)
    #argument for noise mean
    parser.add_argument("-m", "--mean", help="noise mean for generating distribution (default is 0)", type=float, default=0.0)

    #argument for noise distribution type
    parser.add_argument("-d", "--distribution", help="noise generating distribution. (default is constant)"
                        "normal: normal distribution"
                        "uniform: uniform distribution"
                        "constant: value is equal to the chosen mean"
                        , choices=NOISE_GENERATING_DISTS, default="constant")
    #argument for noise operation type
    parser.add_argument("-o", "--operation", help="noise operation type (how will be noise added to the spectogram). (default is additive)"
                        "additive: add noise to the spectogram"
                        "multiplicative: multiply the spectogram with noise (noise values are coefficients)"
                        , choices=NOISE_OPERATION_TYPES, default="additive")
    #argument for noise scope
    parser.add_argument("-s", "--scope", help="noise scope. (default is pixel)"
                        "pixel: add noise to each pixel in the spectogram"
                        "column: each column in the spectogram will have the same noise, but different from other columns"
                        "row: each row in the spectogram will have the same noise, but different from other rows"
                        "entire_picture: the entire spectogram will have the same noise"
                        , choices=NOISE_SCOPE, default="pixel")

    return parser

def plot_losses(losses_all, output_path):
    '''
    Plots the losses of the model during training and saves it to the output_path

    params:
        losses_all - list of losses (total loss, rec loss, kl loss) for each epoch
        output_path - path to save the plot
    '''
    losses = [loss[0] for loss in losses_all]
    rec_losses = [loss[1] for loss in losses_all]
    kl_losses = [loss[2] for loss in losses_all]

    plt.plot(losses, label='total loss')
    plt.plot(rec_losses, label='rec loss')
    plt.plot(kl_losses, label='kl loss')
    plt.title('Train loss in each epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')  # Add this line to set y-axis to logarithmic scale
    plt.legend()

    plt.savefig(os.path.join(output_path, 'train-loss.png'))


def main(argv: Sequence[str] | None =None) -> None:
    parser = parse_arguments()
    args = parser.parse_args(argv)

    if args.scaler == 'None':
        args.scaler = None

    #start timer
    start = datetime.datetime.now()

    #create output directory
    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)

    #create log file
    sys.stdout = open(os.path.join(args.output_path, f'training.log'), 'a+')

    print(f'Date and time of training: {start.strftime("%Y-%m-%d %H:%M:%S")}')
    
    #choose sample groups
    if args.sample_group == 'all':
        args.sample_group = [group for group in os.listdir(args.data_dir) if os.path.isdir(os.path.join(args.data_dir, group))]
    else:
        args.sample_group = args.sample_group.split(',')

    #prepare scaler 
    if args.scaler:
        scaler = create_scaler(args.scaler)
        print(f'Scaler created with config: \n'
                f'\tScaler type: {args.scaler}\n')
        args.scaler = scaler
    else:
        scaler = None
        print('No scaler used.\n')


    #prepare data loader
    train_loader = prepare_train_loader(args.data_dir, args.sample_group, length=args.pad_or_trim_length, batch_size=args.batch_size, scaler=scaler)
    print(f'Data prepared for training. Sample groups: {", ".join(args.sample_group)}\n, mfcc length: {args.pad_or_trim_length}, data directory {args.data_dir}')

    #device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device : {device}\n')

    #create model
    model = create_model(args.model, args.latent_dim).to(device)
    print(f'Model {args.model_name} created. (model type: {args.model})\n')

    #noise function
    if args.noise:
        noise_function = generate_noise(args.mean, args.variance, args.distribution, args.scope, args.operation)
        print(f'Noise function created with config: \n'
                f'\tNoise distribution: {args.distribution}\n'
                f'\tNoise operation: {args.operation}\n'
                f'\tNoise scope: {args.scope}\n'
                f'\tNoise variance: {args.variance}\n'
                f'\tNoise mean: {args.mean}\n')

    else:
        noise_function = lambda x:x
        print('No noise added to the spectograms.\n')
                
    #train the model
    losses = train(model, train_loader, args.epochs, device, log_file, noise_function=noise_function, kl_regularisation=args.kl_regularisation, learning_rate=args.learning_rate)

    #save model
    torch.save(model.state_dict(), os.path.join(args.output_path, f'model.pkl'))
    print(f'Model saved to {args.output_path}')

    plot_losses(losses, args.output_path)

    # save config
    save_config(args.output_path, args, MFCC_KWARGS)

    #end timer
    end = datetime.datetime.now()
    print(f'Training finished in {end-start}')


if __name__ == '__main__':
    main()
