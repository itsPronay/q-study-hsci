import torch 
import torch.nn as nn
import argparse
import wandb
import os 
import scipy.io as sio
from spectralFormer.vit_pytorch import ViT
from spectralSpacialMamba.run import run as run_mamba

parser = argparse.ArgumentParser(description='SpectralFormer')
parser.add_argument('--model', type=str,choices=['SpectralFormer', 'SpectralSpacialMamba'], default='SpectralSpacialMamba')
parser.add_argument('--dataset', type=str, choices=['UP', 'NF', 'HC', 'P', 'Houston'], default='UP')
# parser.add_argument('--group_size', type=int, default=145)
# parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--patches', type=int, default=5)
parser.add_argument('--band_patch', type=int, default=9)

# wandb args
parser.add_argument("--wandb_mode", default="online", choices=["online", "offline", "disabled"])
parser.add_argument('--project_name', type=str, default='Quantization Study')

# quantization specific args
parser.add_argument('--nbits', type=int, default=8)
parser.add_argument('--group_size', type=lambda x: None if x.lower() == 'none' else int(x), default=64)
parser.add_argument('--del_orig', action='store_true', help='if True, delete the original Linear weight inside HQQLinear')
parser.add_argument('--verbose', action='store_true', help='if True, print replacement information')
parser.add_argument('--exclude_names', nargs='*', default=[], help='List of module names to exclude from quantization, e.g., "blocks.11.mlp.fc1 blocks.11.mlp.fc2"')

# these are the args for spectralformer specifically, 
# Pre training
parser.add_argument('--train_num', type=int, default=20)
parser.add_argument('--train_loop', type=int, default=1)
parser.add_argument('--windowsize', type=int, default=27) # 13 * 2 + 1 = 27
parser.add_argument('--type', type=str, default='none')

# training parameter for mamba
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--drop_rate', type=float, default=0.0)
parser.add_argument('--lr_decay', type=float, default=0.5)

# model parameter
parser.add_argument('--model_id', type=int, default=2,help='0: 1D, 1: 2D, 2: SS')

parser.add_argument('--spe_windowsize', type=int, default=3)
parser.add_argument('--spa_patch_size', type=int, default=3)
parser.add_argument('--spe_patch_size', type=int, default=2)
parser.add_argument('--hid_chans', type=int, default=64)
parser.add_argument('--embed_dim', type=int, default=64)
parser.add_argument('--depth', type=int, default=4)

parser.add_argument('--use_bi', default=True, type=lambda x: (str(x).lower() == 'true'), help='use bidirection or not' )  
parser.add_argument('--use_global', default=True, type=bool,help='use token meaning or not') 
parser.add_argument('--use_cls', default=True, type=bool,help='use class tken or not') 
parser.add_argument('--use_fu', default=True, type=lambda x: (str(x).lower() == 'true'),help='use center augmentation fusion or not') 

args = parser.parse_args()

def main():
    if not os.path.exists('classification_maps'):
        os.makedirs('classification_maps')
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







