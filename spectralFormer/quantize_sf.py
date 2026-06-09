import torch
from hqq.core.quantize import BaseQuantizeConfig
import hqq_wrapper
from spectralSpacialMamba.utils import test_batch
import numpy as np
from utils.get_model_summary import getParamCount

def test_batch_quantized(args, model):
    exclude_layers = [
        "cls_head"
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

    #check if model has been quantized
    print("\n[INFO] Model after quantization:")
    getParamCount(model, printLayers=True)

    return model