import torch
from hqq.core.quantize import BaseQuantizeConfig
import hqq_wrapper
from spectralSpacialMamba.utils import test_batch
import numpy as np
from utils.get_model_summary import getParamCount, print_quantization_summary, printWeightStatistics


def test_batch_quantized(args, model, image, index, BATCH_SIZE, nTrain_perClass, nvalid_perClass, halfsize,):
    getParamCount(model, printLayers=True)

    MAMBA_EXCLUDE_LAYERS = [
        "dt_proj",    # directly accesses .weight in forward_core line 248, so had to exclude
        "linear",     # 299x299 = 89401, not divisible by any group size, excluded 
        "head",       # classification head
    ]


    model = hqq_wrapper.replace_all_linear_with_hqq_safe(
        model=model,
        nbits=args.nbits,
        group_size=args.group_size,
        compute_dtype=torch.float32,  # must match your model's dtype
        del_orig=args.del_orig,
        verbose=args.verbose,
        exclude_names=MAMBA_EXCLUDE_LAYERS,
    )

      # check if model has been quantized
    if args.print_quantization_summary:
        print("\n[INFO]__________________________________ Model after quantization: __________________________________")
        getParamCount(model, printLayers=args.print_layers)
        print_quantization_summary(model)

    # test quantized model
    model.eval()

    true_cla_q, oa_q, aa_q, kappa_q, cm_q, pred_q = test_batch(
        model, image, index, BATCH_SIZE, nTrain_perClass, nvalid_perClass, halfsize
    )

    print('quantized overall_accuracy: {0:f}'.format(oa_q))
    print('quantized average_accuracy: {0:f}'.format(aa_q))
    print('quantized kappa:{0:f}'.format(kappa_q))

    return true_cla_q, oa_q, aa_q, kappa_q, cm_q, pred_q
    
