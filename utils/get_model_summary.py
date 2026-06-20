from pyexpat import model

import numpy as np
import torch
from hqq.core.quantize import HQQLinear
import torch.nn as nn


def getParamCount(model, printLayers=False):
    total_param = 0
    print("\n[INFO] __________________________________________Printing all Layers with Param size:__________________________________________")
    for name, param in model.named_parameters():
        if param.requires_grad:
            num_param = np.prod(param.size())
            if param.dim() > 1 and printLayers:
                print(name+':', 'x'.join(str(x) for x in list(param.size())), '=', num_param)
            elif printLayers:
                print(name+':', num_param)
            total_param += num_param
    
    print("\nTotal Trainable Parameters:", total_param)
    if printLayers:
        printAllLinearLayers(model)
    return total_param

def printAllLinearLayers(model):
    print("\n[INFO] __________________________________________Printing all Linear Layers:__________________________________________")
    total_linear_layers = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            print(name, module.weight.shape)
            total_linear_layers += 1
    print(f"\nTotal Linear layers: {total_linear_layers}")
    print("__________________________________________________________________________________________________________________________")


def printWeightStatistics(model):
    print("\n[INFO] __________________________________________Model Weight Statistics:__________________________________________")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: mean={param.data.mean():.4f}, std={param.data.std():.4f}, min={param.data.min():.4f}, max={param.data.max():.4f}")


def print_quantization_summary(model):
    total_layers = 0
    quantized_layers = 0
    
    for name, module in model.named_modules():
        if isinstance(module, HQQLinear):
            nbits = module.quant_config['weight_quant_params']['nbits']
            group_size = module.quant_config['weight_quant_params']['group_size']
            shape = module.W_q.shape
            print(f"{name}: {nbits}-bit | group_size={group_size} | W_q shape={shape}")
            quantized_layers += 1
        elif isinstance(module, torch.nn.Linear):
            print(f"{name}: NOT quantized (fp32) | shape={module.weight.shape}")
        total_layers += 1
    
    print(f"\nQuantized: {quantized_layers} layers, Total: {total_layers} layers, Quantization Ratio: {quantized_layers/total_layers:.2%}")

    # print_quantization_summary(model)