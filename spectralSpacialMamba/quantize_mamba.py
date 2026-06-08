import torch
from hqq.core.quantize import BaseQuantizeConfig
import hqq_wrapper
from spectralSpacialMamba.utils import test_batch
import numpy as np


def getParamCount(model, printLayers=False):
    total_param = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            num_param = np.prod(param.size())
            if param.dim() > 1 and printLayers:
                print(name+':', 'x'.join(str(x) for x in list(param.size())), '=', num_param)
            elif printLayers:
                print(name+':', num_param)
            total_param += num_param
    
    print("\nTotal Trainable Parameters:", total_param)
    return total_param


def test_batch_quantized(args, model, image, index, BATCH_SIZE, nTrain_perClass, nvalid_perClass, halfsize,):
    getParamCount(model, printLayers=True)

    MAMBA_EXCLUDE_LAYERS = [
        "dt_proj",    # directly accesses .weight in forward_core line 248
        "x_proj",     # same issue
        "out_proj",   # same issue
        "linear",     # 299x299 = 89401, not divisible by any group size
        "head",       # classification head
    ]
        
    quant_config = BaseQuantizeConfig(
        nbits=args.nbits,
        group_size=args.group_size,
    )

# put all these in args
    # model = hqq_wrapper.replace_all_linear_with_hqq_safe(
    #     model = model,
    #     quant_config = quant_config,
    #     compute_dtype = torch.float32,
    #     del_orig = args.del_orig,
    #     verbose = args.verbose,
    #     exclude_names = MAMBA_EXCLUDE_LAYERS,
    # )

    model = hqq_wrapper.replace_all_linear_with_hqq_safe(
        model=model,
        nbits=args.nbits,
        group_size=args.group_size,
        # device=device,
        exclude_names=MAMBA_EXCLUDE_LAYERS,
    )

    model.eval()

    true_cla_q, oa_q, aa_q, kappa_q, cm_q, pred_q = test_batch(
        model, image, index, BATCH_SIZE, nTrain_perClass, nvalid_perClass, halfsize
    )

    print('quantized overall_accuracy: {0:f}'.format(oa_q))
    print('quantized average_accuracy: {0:f}'.format(aa_q))
    print('quantized kappa:{0:f}'.format(kappa_q))

    return true_cla_q, oa_q, aa_q, kappa_q, cm_q, pred_q
    




        
