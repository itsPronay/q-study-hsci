from pyexpat import model

import numpy as np
import torch
from hqq.core.quantize import HQQLinear
import torch.nn as nn
import numpy as np
import time
import torch


def measure_latency_throughput(model, test_loader, device, is_quantized=False, warmup=10):
    """
    Runs inference over the entire test_loader.
    Latency = total_time / total_samples  * 1000 (milliseconds per sample)
    Throughput = total_samples / total_time  (samples per second)
    """
    model.eval()
    model = model.to(device)

    total_samples = 0

     # warmup
    with torch.no_grad():
        for i, (x, _) in enumerate(test_loader):
            if i >= warmup:
                break
            x = x.to(device)
            model(x)

    if device.type == 'cuda':
        torch.cuda.synchronize()

    start = time.perf_counter()

    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            model(inputs)
            total_samples += inputs.size(0)

    if device.type == 'cuda':
        torch.cuda.synchronize()

    total_time_sec = time.perf_counter() - start

    latency_ms         = (total_time_sec / total_samples) * 1000   # ms per sample
    throughput         = total_samples / total_time_sec              # samples / sec

    print (f"Total samples: {total_samples}, Total time: {total_time_sec:.4f} sec, Latency: {latency_ms:.4f} ms/sample, Throughput: {throughput:.2f} samples/sec")
    
    if is_quantized:
        prefix = "quant_"
    else:
        prefix = ""

    return {
        f'{prefix}_latency_ms':       round(latency_ms, 2),
        f'{prefix}_throughput_samples_per_sec': round(throughput, 2),
    }


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
    return total_param

