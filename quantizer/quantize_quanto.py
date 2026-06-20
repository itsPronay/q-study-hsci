from optimum.quanto import (
    QTensor,
    freeze,
    qint8,
    qint4,
    qfloat8,
    quantize,
)

def quanto_quantization(args, model):

    qtype_map = {
        "4":  qint4,
        "8":  qint8,
        # "8": qfloat8,
    }

    weights_qtype = qtype_map[args.nbits]

    # device = next(model.parameters()).device

    # quantize weights only (activations=None)
    quantize(
        model,
        weights=weights_qtype,
        activations=None,
        exclude="head",
    )
    
    # Step 2: freeze — replace float weights with QTensors
    freeze(model)

    return model