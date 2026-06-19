import torch
from hqq.core.quantize import BaseQuantizeConfig
import quantizer.hqq_wrapper as hqq_wrapper
from spectralSpacialMamba.utils import test_batch
import numpy as np
from utils.get_model_summary import getParamCount, print_quantization_summary, printWeightStatistics


def quantize_model(args, model):
    # check if model has been quantized
    if args.print_quantization_summary:
        print("\n[INFO]__________________________________ Printing model summary - before quantization __________________________________")
        getParamCount(model, printLayers=args.print_layers)

    if args.model == 'mvit':
        exclude_layers = [
            "cls_head"
        ]
    elif args.model == 'SpectralFormer':
        exclude_layers = [
            "mlp_head",
            "patch_to_embedding", # fails for group_size = 64
            # "net.0",               # FFN layer 1 - too small for group_size=512
            # "net.3",               # FFN layer 2 - too small for group_size=512
        ]
    elif args.model == 'SpectralSpacialMamba':
        exclude_layers = [
            "dt_proj",    # directly accesses .weight in forward_core line 248, so had to exclude
            "linear",     # 299x299 = 89401, not divisible by any group size, excluded 
            "head",       # classification head
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