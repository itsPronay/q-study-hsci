from torchao.quantization import quantize_, Int4WeightOnlyConfig, Int8WeightOnlyConfig
import torch

def torchao_quantization(args, model):
    def filter_fn(module, fqn):
        exclude = ['cls_head']
        return isinstance(module, torch.nn.Linear) and fqn not in exclude

    if args.nbits == 4:
        config = Int4WeightOnlyConfig(group_size=args.group_size)
    elif args.nbits == 8:
        config = Int8WeightOnlyConfig(group_size=args.group_size)

    quantize_(model, config, filter_fn=filter_fn)
    return model