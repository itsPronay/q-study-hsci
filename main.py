from matplotlib.pylab import shape
import torch 
import torch.nn as nn
import argparse
import wandb
import os 
import scipy.io as sio
# from spectralFormer.train import run as run_spectralformer
from spectralSpacialMamba.run import run as run_mamba
from mvit.run import run_mvit
from spectralFormer.train import train_spectralformer


parser = argparse.ArgumentParser(description='Study quantization of different models on hyperspectral image classification')

parser.add_argument('--wandb_project', type=str, default='Quantization Study', help='wandb project name')

parser.add_argument('--model', type=str,choices=['SpectralFormer', 'SpectralSpacialMamba', 'mvit'], default='SpectralSpacialMamba')
parser.add_argument('--dataset', type=str, choices=['UP', 'NF', 'HC', 'Pavia', 'Indian'], default='UP')
# parser.add_argument('--group_size', type=int, default=145)
# parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--patches', type=int, default=5)
parser.add_argument('--band_patch', type=int, default=9)
parser.add_argument('--train_num', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=512)
# wandb args
parser.add_argument("--wandb_mode", default="online", choices=["online", "offline", "disabled"])
parser.add_argument('--project_name', type=str, default='Quantization Study')

# quantization specific args
parser.add_argument('--nbits', type=int, default=8)
parser.add_argument('--print_quantization_summary', type=int, default=1) # 0 for false, 1 for true
parser.add_argument('--group_size', type=lambda x: None if x.lower() == 'none' else int(x), default=64)
parser.add_argument('--del_orig', action='store_true', help='if True, delete the original Linear weight inside HQQLinear')
parser.add_argument('--verbose', action='store_true', help='if True, print replacement information')
parser.add_argument('--exclude_names', nargs='*', default=[], help='List of module names to exclude from quantization, e.g., "blocks.11.mlp.fc1 blocks.11.mlp.fc2"')


# these are the args for spectralSpacialMamba specifically, 
# Pre training
parser.add_argument('--train_loop', type=int, default=1)
parser.add_argument('--windowsize', type=int, default=27) # 13 * 2 + 1 = 27
parser.add_argument('--type', type=str, default='none')

# training parameter for mamba
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


#MViT specific args
# parser.add_argument('--dataset', choices=['LongKou', 'HanChuan', 'HongHu', 'Pavia'], default='Pavia', help='dataset to use')
# parser.add_argument('--seed', type=int, default=42, help='number of seed')
# parser.add_argument('--batch_size', type=int, default=30, help='number of batch size')
parser.add_argument('--patch_size_mvit', type=int, default=15, help='size of patches')
# parser.add_argument('--epoches', type=int, default=100, help='epoch number')
parser.add_argument('--learning_rate_mvit', type=float, default=1e-3, help='learning rate')
parser.add_argument('--gamma_mvit', type=float, default=0.99, help='gamma')
parser.add_argument('--weight_decay_mvit', type=float, default=0.001, help='weight_decay')
# parser.add_argument('--train_number_mvit', type=int, default=25, help='num_train_per_class')

#SpectralFormer specific args
# parser.add_argument('--dataset', choices=['Indian', 'Pavia', 'Houston'], default='Indian', help='dataset to use')
parser.add_argument('--flag_test', choices=['test', 'train'], default='train', help='testing mark')
parser.add_argument('--mode', choices=['ViT', 'CAF'], default='CAF', help='mode choice')
parser.add_argument('--gpu_id', default='0', help='gpu id')
# parser.add_argument('--seed', type=int, default=0, help='number of seed')
# parser.add_argument('--batch_size', type=int, default=64, help='number of batch size')
parser.add_argument('--test_freq', type=int, default=5, help='number of evaluation')
parser.add_argument('--patches_sf', type=int, default=1, help='number of patches')
parser.add_argument('--band_patches_sf', type=int, default=1, help='number of related band')
# parser.add_argument('--epoches', type=int, default=300, help='epoch number')
parser.add_argument('--learning_rate_sf', type=float, default=5e-4, help='learning rate')
parser.add_argument('--gamma_sf', type=float, default=0.9, help='gamma')
parser.add_argument('--weight_decay_sf', type=float, default=0, help='weight_decay')

args = parser.parse_args()

def main():
    if not os.path.exists('classification_maps'):
        os.makedirs('classification_maps')
        
    if args.wandb_mode != 'disabled':
        wandb.init(
            project = args.wandb_project,
            name = f"{args.model}_{args.dataset}_quantization_{args.nbits}bits_group{args.group_size}",
            mode = args.wandb_mode,
            config = vars(args)
        )

    if args.model == 'SpectralFormer':
        train_spectralformer(args)
    elif args.model == 'SpectralSpacialMamba':
        run_mamba(args)
    elif args.model == 'mvit':
        run_mvit(args)
    else:
        raise ValueError(f"Unknown model: {args.model}")

if __name__ == '__main__':
    main()







