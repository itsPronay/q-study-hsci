import torch
from hqq.core.quantize import BaseQuantizeConfig
import hqq_wrapper
from spectralSpacialMamba.utils import test_batch
import numpy as np
from utils.get_model_summary import getParamCount, printWeightStatistics, print_quantization_summary


def test_batch_quantized(args, model):
    exclude_layers = [
        "mlp_head",
        # "patch_to_embedding" # fails for group_size = 64
    ]

    # getParamCount(model, printLayers=True)        
    quant_config = BaseQuantizeConfig(
        nbits=args.nbits,
        group_size=args.group_size,
    )

    model = hqq_wrapper.replace_all_linear_with_hqq(
        model=model,
        quant_config=quant_config,
        compute_dtype=torch.float32, 
        del_orig=args.del_orig,
        verbose=args.verbose,
        exclude_names=exclude_layers,
    )

    if args.print_quantization_summary:
        print("\n[INFO]__________________________________ Model after quantization: __________________________________")
        getParamCount(model, printLayers=args.print_layers)
        print_quantization_summary(model)

    return model