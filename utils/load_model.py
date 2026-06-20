from models.mvit import MViT
from models.spectralFormer import ViT
from models.massFormer import Massformer
from models.spectralSpacialMamba.model import mamba_1D_model, mamba_2D_model, mamba_SS_model
import torch.nn as nn

def model_loader(args, num_class):
    if args.model == 'mvit':
        model = MViT(num_classes = num_class).cuda()
    elif args.model == 'sf':
        model = ViT(
            img_size = args.patch_size,
            in_chans = args.pca_band,
            num_classes = num_class,
            dim = 64,
            depth = 5,
            heads = 4,
            mlp_dim = 8,
            dropout = 0.1,
            emb_dropout = 0.1,
        ).cuda()
    elif args.model == 'ssm':
        model = mamba_SS_model(
            spa_img_size=(27, 27),
            spe_img_size=(3,3), 
            spa_patch_size=args.patch_size, 
            spe_patch_size=args.band_patch, 
            in_chans=args.pca_band, 
            hid_chans = 64, 
            embed_dim=64, 
            drop_path=0.0, 
            nclass=num_class, 
            depth=4, 
            bi=True,
            norm_layer=nn.LayerNorm, 
            global_pool=True, 
            cls = True, 
            fu = True
        ).cuda()
    elif args.model == 'mf':
        model = Massformer(num_classes=num_class).cuda()
    else:
        raise Exception('model name could not found')
    
    return model
