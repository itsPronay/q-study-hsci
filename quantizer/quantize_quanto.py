from optimum.quanto import quantize, freeze, qint8, qint4, qint2
import torch

NBITS_MAP = {
    8: qint8,
    4: qint4,
    2: qint2
}

def get_all_exclude_layers(model, also_exclude_names=None):
    if also_exclude_names is None:
        also_exclude_names = []

    exclude_layer_names = list(also_exclude_names)  # start with manually excluded layers

    # quanto only supports Linear, Conv2d, and LayerNorm
    # so getting all other layers to exclude them from quantization
    for name, module in model.named_modules():
        if name == '':  # skip root module
            continue
        if not isinstance(module, (torch.nn.Linear, torch.nn.Conv2d, torch.nn.LayerNorm)):
            exclude_layer_names.append(name)

    return exclude_layer_names


def quanto_quantization(args, model):
    exclude_layers = []

    if args.model == 'mvit':
        exclude_layers = [
            "cls_head",
            "conv2d.0"
        ]

    exclude_layers = get_all_exclude_layers(model, also_exclude_names=exclude_layers)

    qtype = NBITS_MAP.get(args.nbits)

    quantize(model, weights=qtype, exclude=exclude_layers)
    # freeze(model)
    return model