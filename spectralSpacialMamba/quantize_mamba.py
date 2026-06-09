import torch
from hqq.core.quantize import BaseQuantizeConfig
import hqq_wrapper
from spectralSpacialMamba.utils import test_batch
import numpy as np
from utils.get_model_summary import getParamCount, print_quantization_summary, printWeightStatistics


def test_batch_quantized(args, model, image, index, BATCH_SIZE, nTrain_perClass, nvalid_perClass, halfsize,):
    getParamCount(model, printLayers=True)

    MAMBA_EXCLUDE_LAYERS = [
        "dt_proj",    # directly accesses .weight in forward_core line 248
        # "x_proj",     # same issue
        # "out_proj",   # same issue
        "linear",     # 299x299 = 89401, not divisible by any group size
        "head",       # classification head
    ]

    # quant_config=BaseQuantizeConfig(
    #         nbits=args.nbits,
    #         group_size=args.group_size,
    #     )
    
    # model = hqq_wrapper.replace_all_linear_with_hqq(
    #     model=model,
    #     quant_config=quant_config,
    #     compute_dtype=torch.float32, 
    #     del_orig=args.del_orig,
    #     verbose=args.verbose,
    #     exclude_names=MAMBA_EXCLUDE_LAYERS,
    # )

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
    



        
#  return self._call_impl(*args, **kwargs)
#            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1787, in _call_impl
#     return forward_call(*args, **kwargs)
#            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/content/q-study-hsci/spectralSpacialMamba/model.py", line 381, in forward
#     x = self.forward_features(x)
#         ^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/content/q-study-hsci/spectralSpacialMamba/model.py", line 371, in forward_features
#     x_spa, x_spe = blk(x_spa, x_spe)
#                    ^^^^^^^^^^^^^^^^^
#   File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1776, in _wrapped_call_impl
#     return self._call_impl(*args, **kwargs)
#            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1787, in _call_impl
#     return forward_call(*args, **kwargs)
#            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/content/q-study-hsci/spectralSpacialMamba/model.py", line 89, in forward
#     x_spa = self.spa_block(x_spa)   #(N, HW/P^2, D)
#             ^^^^^^^^^^^^^^^^^^^^^
#   File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1776, in _wrapped_call_impl
#     return self._call_impl(*args, **kwargs)
#            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1787, in _call_impl
#     return forward_call(*args, **kwargs)
#            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/content/q-study-hsci/spectralSpacialMamba/model.py", line 34, in forward
#     x1 = self.self_attention(x)
#          ^^^^^^^^^^^^^^^^^^^^^^
#   File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1776, in _wrapped_call_impl
#     return self._call_impl(*args, **kwargs)
#            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1787, in _call_impl
#     return forward_call(*args, **kwargs)
#            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/content/q-study-hsci/spectralSpacialMamba/mamba.py", line 270, in forward
#     y = self.forward_core(x)
#         ^^^^^^^^^^^^^^^^^^^^
#   File "/content/q-study-hsci/spectralSpacialMamba/mamba.py", line 248, in forward_core
#     dt = self.dt_proj.weight @ dt.t()