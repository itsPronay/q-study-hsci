# Q-Study HSCI

This project trains hyperspectral image classification models, evaluates the saved checkpoint, and optionally quantizes the model with HQQ or Quanto.

## Quick Start

1. Train a model first.

```bash
python train.py --model mvit --dataset UP
```

2. Evaluate the saved model and quantize it.

```bash
python eval.py --model mvit --dataset UP --quant_method hqq --nbits 8 --group_size 64
```

If you want to compare the original and quantized model with CKA, enable the `--cka` flag:

```bash
python eval.py --model mvit --dataset UP --quant_method hqq --nbits 8 --group_size 64 --cka True
```

## Notes

- `train.py` contains the training arguments.
- `eval.py` contains evaluation, quantization, and comparison arguments.
- For Indian Pines, the training split is capped at 10 samples per class in the code.