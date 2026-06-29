# from pyexpat import model

import numpy as np
# import torch
# from hqq.core.quantize import HQQLinear
# import torch.nn as nn
# import numpy as np
# import time
# import torch
# import numpy as np

# from sklearn.preprocessing import StandardScaler

# import torch
# import torch.utils.data as Data
# import wandb
# from utils.data_prepare import mirror_hsi
# from utils.data_prepare import choose_train_and_test
# from utils.data_prepare import train_and_test_data, train_and_test_label
# from utils.data_prepare import applyPCA
# from utils.download_dataset import downloadAndLoadDataset
# from utils.load_model import model_loader
# from quantizer.quantize_hqq import hqq_quantization
# from quantizer.quantize_quanto import quanto_quantization

# from utils.load_model import model_loader
# import argparse

# parser = argparse.ArgumentParser('Latency and Throughput Measurement')

# parser.add_argument('--model', type=str)
# parser.add_argument('--group_size', type=lambda x: None if x.lower() == 'none' else int(x))
# parser.add_argument('--nbits', type=str)
# parser.add_argument('--warmup', type=int, default=2)
# parser.add_argument('--runs', type=int, default=5)
# parser.add_argument('--quant_method', type=str)
# parser.add_argument('--dataset', type=str, choices=['UP', 'NF', 'HC', 'Pavia', 'Indian', 'Houston'], default='UP')

# parser.add_argument('--batch_size', type=int, default=64)
# parser.add_argument('--patch_size', type=int, default=15)
# parser.add_argument('--band_patch', type=int, default=1)
# parser.add_argument('--pca_band', type=int, default=30)
# parser.add_argument('--train_num', type=int, default=20)
# parser.add_argument('--seed', type=int, default=42)

# parser.add_argument('--del_orig', type=lambda x: x.lower() == 'true', default=True, help='if True, delete the original Linear weight inside HQQLinear')
# parser.add_argument('--verbose', type=lambda x: x.lower() == 'true', default=True, help='if True, print replacement information')

# parser.add_argument("--wandb_mode", default="online", choices=["online", "offline", "disabled"])
# parser.add_argument('--wandb_project', type=str, default='QHSICdyf', help='wandb project name')

# args = parser.parse_args()

# models = ['sf', 'ssm', 'mvit', 'mf']
# hqq_nbits = [1, 2, 3, 4, 8]
# quanto_nbits = [2, 4, 8, 88]
# group_sizes = [8, 16, 32, 64]

# def setupWandb(args):
#     if args.model == 'mvit':
#         model_name = 'MViT'
#     elif args.model == 'ssm':
#         model_name = 'SpectralSpacialMamba'
#     elif args.model == 'sf':
#         model_name = 'SpectralFormer'
#     elif args.model == 'mf':
#         model_name = 'MassFormer'
#     if args.nbits is not None:
#         run_name = f"{model_name}_{args.quant_method}_nbits{args.nbits}_group{args.group_size}"
#     else:
#         run_name = f"{model_name}_BASE"

#     if args.wandb_mode != 'disabled':
#         wandb.init(
#             project = args.wandb_project,
#             name = run_name,
#             mode = args.wandb_mode,
#             config = vars(args)
#         )

# def measure_latency_throughput(model, test_loader, is_quantized=False, warmup=10, runs=10):
#     """
#     Runs inference over the entire test_loader.
#     Latency = total_time / total_samples  * 1000 (milliseconds per sample)
#     Throughput = total_samples / total_time  (samples per second)
#     """

#     device = next(model.parameters()).device  # follows model — cpu or cuda automatically

#     # warmup
#     with torch.no_grad():
#         for i, (x, _) in enumerate(test_loader):
#             if i >= warmup:
#                 break
#             x = x.to(device)
#             model(x)

#     if device.type == 'cuda':
#         torch.cuda.synchronize()

#     latencies = []
#     total_samples = 0

#     for run in range(runs):
#         run_samples = 0
#         start = time.perf_counter()
#         with torch.no_grad():
#             for x, _ in test_loader:
#                 x = x.to(device)
#                 _ = model(x)
#                 run_samples += x.size(0)
#         if device.type == 'cuda':
#             torch.cuda.synchronize()
#         elapsed = (time.perf_counter() - start) * 1000  # ms
#         latencies.append(elapsed)
#         total_samples = run_samples  # same every run

#     total_time_sec = np.mean(latencies) / 1000

#     latency_ms  = np.mean(latencies)
#     latency_std = np.std(latencies)
#     latency_min = np.min(latencies)
#     latency_max = np.max(latencies)
#     throughput  = total_samples / total_time_sec  # samples / sec

#     print(f"Total samples: {total_samples}, Total time: {total_time_sec:.4f} sec, Latency: {latency_ms:.4f} ms/run, Throughput: {throughput:.2f} samples/sec")
#     print(f"Latency stats — mean: {latency_ms:.2f} ms | std: {latency_std:.2f} ms | min: {latency_min:.2f} ms | max: {latency_max:.2f} ms")

#     if is_quantized:
#         prefix = "quant_"
#     else:
#         prefix = ""

#     return {
#         'device': device.type,
#         f'{prefix}latency_ms':                 round(latency_ms, 2),
#         f'{prefix}latency_std_ms':             round(latency_std, 2),
#         f'{prefix}throughput_samples_per_sec': round(throughput, 2),
#     }


# def log_latency_throughput_to_wandb():
#     print(args)
#     # Load data
#     data, label = downloadAndLoadDataset(args.dataset)
#     num_classes = int(np.max(label))

#     # apply normalization
#     shapeor = data.shape
#     data = data.reshape(np.prod(data.shape[:2]), np.prod(data.shape[2:]))

#     std_scaler = StandardScaler()
#     std_data = std_scaler.fit_transform(data)
#     data = std_data.reshape(shapeor)

#     data, pca = applyPCA(data, numComponents=args.pca_band)

#     # data size
#     height, width, band = data.shape
#     print("height={0}, width={1}, band={2}".format(height, width, band))

#     mirror_data = mirror_hsi(height, width, band, data, patch_size=args.patch_size)

#     if args.dataset == 'Indian': #hardcoding cause, a class in indian Pines has only 
#         train_num = 10
#     else:
#         train_num= args.train_num
#     total_pos_train, total_pos_test, total_pos_valid, number_train, number_test, number_valid = choose_train_and_test(
#         label, num_train_per_class=train_num, seed=args.seed
#     )

#     _, x_test, _ = train_and_test_data(
#         mirror_data, band, total_pos_train, total_pos_test, total_pos_valid, patch_size=args.patch_size
#     )
#     _, y_test, _ = train_and_test_label(number_train, number_test, number_valid, num_classes)

#     x_test = torch.from_numpy(x_test.transpose(0, 3, 1, 2)).type(torch.FloatTensor) 
#     if args.model == 'mvit' or args.model == 'mf':
#         x_test = x_test.unsqueeze(1)
#     print(x_test.shape)
#     y_test = torch.from_numpy(y_test).type(torch.LongTensor)  
#     test_label = Data.TensorDataset(x_test, y_test)

#     print("*****************************************************************")
#     print(f"x_test shape: {x_test.shape}")
#     print("*****************************************************************")

#     test_loader = Data.DataLoader(test_label, batch_size=args.batch_size, shuffle=False)

#     for m in models:
#         args.model = m

#         #hqq
#         print("Measuring base for hqq")
#         args.quant_method = 'hqq'
#         model = model_loader(args, num_class=num_classes)
#         base_result = measure_latency_throughput(model, test_loader, is_quantized=False, warmup=args.warmup, runs=args.runs)
        
#         for h in hqq_nbits:
#             for group in group_sizes:
#                 print("Measuring latency and throughput for model: {}, quant_method: {}, nbits: {}, group_size: {}".format(args.model, 'hqq', h, group))

#                 #for hqq: setup first
#                 args.nbits = h
#                 args.group_size = group

#                 setupWandb(args)

#                 quantized_model = hqq_quantization(args, model)
#                 quantized_result = measure_latency_throughput(quantized_model, test_loader, is_quantized=True, warmup=args.warmup, runs=args.runs)
#                 if args.wandb_mode != 'disabled':
#                     wandb.log(base_result)
#                     wandb.log(quantized_result)  
#                     wandb.finish()
        
#         # to keep the results fair, we again revaluating , but this time in CPU
#         print("Measuring base for quanto")
#         args.quant_method = 'quanto'
#         model = model_loader(args, num_class=num_classes)
#         model.cpu()        
#         base_result = measure_latency_throughput(model, test_loader, is_quantized=False, warmup=args.warmup, runs=args.runs)

#         for q in quanto_nbits:
#             if m == 'ssm' and (q == 2 or q == 4):
#                 continue #skip this combination, 
#             args.nbits = q
#             args.group_size = None
#             setupWandb(args)

#             quantized_model = quanto_quantization(args, model)
#             quantized_result = measure_latency_throughput(quantized_model, test_loader, is_quantized=True, warmup=args.warmup, runs=args.runs)
#             if args.wandb_mode != 'disabled':
#                 wandb.log(base_result)
#                 wandb.log(quantized_result)  
#                 wandb.finish()


import torch
from quantizer.hqq_wrapper import find_matching_layers

def find_best_threshold(model, sensitive_layers):
    thresholds = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

    print(f"{'Threshold':>12}  {'Outlier %':>10}")
    print(f"{'─'*30}")

    for t in thresholds:
        pcts = []
        for layer_name in sensitive_layers:
            matches = find_matching_layers(model, layer_name, only_linear=True, print_results=False)
            for _, module in matches:
                W    = module.weight.data.float()
                mean = W.mean()
                std  = W.std()
                pct  = ((W - mean).abs() > t * std).float().mean().item() * 100
                pcts.append(pct)

        avg = sum(pcts) / len(pcts)
        print(f"{t:>11}σ  {avg:>9.2f}%")

def print_outliers(model, layer_names, threshold=3.0):
    model.eval()
    find_best_threshold(model, layer_names)
    
    ignore = ["head", "dt_proj", "mlp_head", "patch_to_embedding", "cls_head"]
    filtered = [l for l in layer_names if l not in ignore]

    print("_"*45)
    print(f"\n[INFO] Outlier analysis for model '{model.__class__.__name__}' with threshold {threshold}σ:")
    print("All the filtered layers to analyze:")
    print(filtered)
    print("_"*45)


    for layer_name in filtered:
        matches = find_matching_layers(model, layer_name, only_linear=True, print_results=False)

        if not matches:
            print(f"\n[INFO] No Linear layers matched '{layer_name}'. Skipping.")
            continue

        print(f"\n[INFO] Outlier analysis for '{layer_name}' — {len(matches)} layer(s) matched:")
        print(f"{'─'*45}")

        for full_name, module in matches:
            W = module.weight.data.float()
            mean = W.mean()
            std  = W.std()

            outlier_mask    = (W - mean).abs() > threshold * std
            outlier_percent = outlier_mask.float().mean().item() * 100

            frobenius = W.norm(p='fro').item()
            rms       = W.pow(2).mean().sqrt().item()
            abs_mean  = W.abs().mean().item()
            abs_max   = W.abs().max().item()

            print(f"  Layer      : {full_name}")
            print(f"  Shape      : {list(W.shape)}")
            print(f"  Mean       : {mean.item():.4f}")
            print(f"  Std        : {std.item():.4f}")
            print(f"  Outliers   : {outlier_percent:.2f}%  (threshold: ±{threshold}σ)")
            print(f"  ── Magnitude ──────────────────────")
            print(f"  Frobenius  : {frobenius:.4f}")
            print(f"  RMS        : {rms:.4f}")
            print(f"  Abs mean   : {abs_mean:.4f}")
            print(f"  Abs max    : {abs_max:.4f}")
            print(f"  {'─'*35}")


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


# if __name__ == "__main__":
#     log_latency_throughput_to_wandb()