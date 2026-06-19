from optimum.quanto import quantize, freeze, qint8, qint4, qint2

NBITS_MAP = {
    8: qint8,
    4: qint4,
    2: qint2
}

def quanto_quantization(args, model):
    exclude_layers = []
    
    if args.model == 'mvit':
        exclude_layers = ['cls_head']
    
    quantize(model, weights=NBITS_MAP.get(args.nbits), exclude=exclude_layers)

    freeze(model)
    return model