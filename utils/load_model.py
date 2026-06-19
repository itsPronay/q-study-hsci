from model.mvit import MViT
from model.spectralFormer import ViT
from model.spectralSpacialMamba.model import mamba_1D_model, mamba_2D_model, mamba_SS_model
import torch.nn as nn

def model_loader(args, num_class):
    if args.model == 'mvit':
        model = MViT(num_class = num_class).cuda()
    elif args.model == 'spectralFormer':
        model = ViT(
            image_size = args.patches_sf,
            near_band = args.band_patches,
            num_patches = args.band,
            num_classes = num_class,
            dim = 64,
            depth = 5,
            heads = 4,
            mlp_dim = 8,
            dropout = 0.1,
            emb_dropout = 0.1,
            mode = 'CAF'
        ).cuda()
    elif args.model == 'spectralSpacialMamba':
        model = mamba_SS_model(
            spa_img_size=(args.windowsize, args.windowsize),
            spe_img_size=(args.spe_windowsize,args.spe_windowsize), 
            spa_patch_size=args.patch_size, 
            spe_patch_size=args.band_patch, 
            in_chans=args.pca_band, 
            hid_chans = args.hid_chans, 
            embed_dim=args.embed_dim, 
            drop_path=args.drop_rate, 
            nclass=num_class, 
            depth=args.depth, 
            bi=args.use_bi,
            norm_layer=nn.LayerNorm, 
            global_pool=args.use_global, 
            cls = args.use_cls, 
            fu = args.use_fu
        ).cuda()
    else:
        raise Exception('model name could not found')
    
    return model
