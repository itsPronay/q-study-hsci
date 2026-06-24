import argparse
import wandb
import torch
import time
import numpy as np
from utils.load_model import model_loader
from eval import get_args
from quantizer.quantize_hqq import hqq_quantization

args = get_args()
# args = parser.parse_args()


def benchmark_model(model, input_tensor, runs=50, warmup_runs=30):
    model.eval()

    # warmup
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(input_tensor)

    # sync before timing
    if input_tensor.device.type == 'cuda':
        torch.cuda.synchronize()

    latencies = []
    start_total = time.perf_counter()
    for _ in range(runs):
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(input_tensor)
        if input_tensor.device.type == 'cuda':
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - start) * 1000)
    total_elapsed = time.perf_counter() - start_total

    batch_size = input_tensor.shape[0]
    total_samples = batch_size * runs
    throughput = total_samples / total_elapsed  # samples per second

    return {
        'mean_latency_ms':            round(float(np.mean(latencies)), 4),
        'min_latency_ms':             round(float(np.min(latencies)), 4),
        'max_latency_ms':             round(float(np.max(latencies)), 4),
        'std_latency_ms':             round(float(np.std(latencies)), 4),
        'throughput_samples_per_sec': round(throughput, 4),
    }


def get_model_size_mb(model):
    """Static size of model parameters + buffers, in MB."""
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / (1024 ** 2)


def get_peak_gpu_memory_mb(model, input_tensor, device):
    """Peak GPU memory allocated during a single forward pass, in MB."""
    if device.type != 'cuda':
        return None
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    model.eval()
    with torch.no_grad():
        _ = model(input_tensor)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated(device) / (1024 ** 2)


def main():
    models     = ["sf", "ssm", "mvit", "mf"]
    quant_bits = [1, 2, 4, 8]
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for m in models:
        for b in quant_bits:
            wandb.init(
                project="Study sdf and throughput of quantized models",
                name=f"{m}_{b}bits",
                mode="online",
                config=vars(args)
            )
            print(f"Running latency test for model: {m} | bits: {b}")
            args.model = m
            args.nbits = b
            model = model_loader(args, num_class=10).to(device)
            model.eval()

            if m in ['mvit', 'mf']:
                input_tensor = torch.randn(1, 1, args.pca_band, args.patch_size, args.patch_size).to(device)
            else:
                input_tensor = torch.randn(1, args.pca_band, args.patch_size, args.patch_size).to(device)

            # ---- FP32 metrics ----
            metrics = benchmark_model(model, input_tensor)
            metrics['model_size_mb'] = round(get_model_size_mb(model), 4)
            metrics['peak_gpu_mem_mb'] = round(get_peak_gpu_memory_mb(model, input_tensor, device), 4)
            print(f"Results — model: {m} | bits: {b} | {metrics}")

            # ---- Quantized metrics ----
            quantized_model = hqq_quantization(args, model).to(device)
            quantized_model.eval()

            quant_metrics = benchmark_model(quantized_model, input_tensor)
            quant_metrics['model_size_mb'] = round(get_model_size_mb(quantized_model), 4)

            # isolate quantized-model-only peak memory (avoid double counting with fp32 model still in memory)
            del model
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            with torch.no_grad():
                _ = quantized_model(input_tensor)
            torch.cuda.synchronize()
            quant_metrics['peak_gpu_mem_mb'] = round(
                torch.cuda.max_memory_allocated(device) / (1024 ** 2), 4
            )

            print(f"Results after quantization — model: {m} | bits: {b} | {quant_metrics}")

            wandb.log({f"fp32_{k}": v for k, v in metrics.items()})
            wandb.log({f"quant_{k}": v for k, v in quant_metrics.items()})
            wandb.finish()


if __name__ == "__main__":
    main()