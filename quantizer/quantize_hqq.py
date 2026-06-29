import torch
from hqq.core.quantize import BaseQuantizeConfig
import quantizer.hqq_wrapper as hqq_wrapper
import numpy as np


def hqq_quantization(args, model):
    exclude_layers = []

    if args.exclude_layers is None:
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
                "head",
            ]
        else:
            raise ValueError(f"Unsupported model {args.model} for quantization")
    else:
        exclude_layers = args.exclude_layers

    print("-" * 50)
    print(f"Quantizing {args.model} with {args.nbits} bits and group-{args.group_size} quantization using HQQ...")
    print(f"Excluding layers from quantization: {exclude_layers}")
    print("-" * 50)

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