from optimum.quanto import (
    freeze,
    qint8,
    qint4,
    qint2,
    qfloat8,
    quantize,
)

def quanto_quantization(args, model):

    exclude_layers = []

    if args.exclude_layers is not None:
        if args.model == 'mvit':
            exclude_layers = [
                "cls_head"
            ]
        elif args.model == 'sf':
            exclude_layers = [
                "mlp_head",
                "patch_to_embedding", 
            ]
        elif args.model == 'ssm':
            exclude_layers = [
                "head",   
            ]
        elif args.model == 'mf':
            exclude_layers = [
                "head",
            ]
    else:
        exclude_layers = args.exclude_layers

    qtype_map = {
        2: qint2,
        4:  qint4,
        8:  qint8,
        88: qfloat8, # pass 88 for float8
    }

    print("-" * 50)
    print(f"Quantizing {args.model} with {args.nbits}-bit quantization using Quanto...")
    print(f"Excluding layers from quantization: {exclude_layers}")
    print("-" * 50)

    weights_qtype = qtype_map[args.nbits]
    
    if args.model != "ssm":
        model = model.cpu()

    # quantize weights only (activations=None)
    quantize(
        model,
        weights=weights_qtype,
        activations=None,
        exclude=exclude_layers,
    )
    
    freeze(model)
    return model