import torch
from hqq.core.quantize import BaseQuantizeConfig
import quantizer.hqq_wrapper as hqq_wrapper
import numpy as np
from utils.get_model_summary import getParamCount, print_quantization_summary, printWeightStatistics


def hqq_quantization(args, model):
    
    if args.model == 'mvit':
        exclude_layers = [
            "cls_head"
        ]
    elif args.model == 'sf':
        exclude_layers = [
            "mlp_head",
            "patch_to_embedding", 
        ]
    elif args.model == 'ssm':
        exclude_layers = [
            "dt_proj",   
            "head",   
        ]
    elif args.model == 'mf':
        exclude_layers = [
            "head"   
        ]
    else:
        raise ValueError(f"Unsupported model {args.model} for quantization")

    model = hqq_wrapper.replace_all_linear_with_hqq_safe(
        model=model,
        nbits=args.nbits,
        group_size=args.group_size,
        compute_dtype=torch.float32, 
        del_orig=args.del_orig,
        verbose=args.verbose,
        exclude_names=exclude_layers,
    )

    return model