import torch 
import torch.nn as nn
import argparse
import wandb
import os 
import scipy.io as sio
from spectralFormer.vit_pytorch import ViT
from spectralSpacialMamba.run import run as run_mamba

parser = argparse.ArgumentParser(description='SpectralFormer')
parser.add_argument('--model', choices=['SpectralFormer', 'SpectralSpacialMamba'], default='SpectralFormer')
parser.add_argument('--dataset', type=str, choices=['UP', 'NF', 'HC', 'P', 'Houston'], default='UP')
parser.add_argument('--group_size', type=int, default=145)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--num_epochs', type=int, default=200)
parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--patches', type=int, default=5)
parser.add_argument('--band_patch', type=int, default=9)

parser.add_argument("--wandb_mode", default="online", choices=["online", "offline", "disabled"])
parser.add_argument('--project_name', type=str, default='Quantization Study')
# parser.add_argument

args = parser.parse_args()

def main():
    # wandb.init(
    #     project = args.project_name,
    #     # name = f"Input shape: {shape}, Device: {args.ai_hub_device}, mode: {args.mode}",
    #     mode = args.wandb_mode,
    #     config = vars(args)
    # )

    if args.model == 'SpectralFormer':
        print('Training SpectralFormer...')
    elif args.model == 'SpectralSpacialMamba':
        run_mamba(args)
    elif args.model == 'mvit':
        print('Training mvit...')
    else:
        raise ValueError(f"Unknown model: {args.model}")

if __name__ == '__main__':
    main()







