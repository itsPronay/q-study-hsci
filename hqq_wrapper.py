import torch
from hqq.core.quantize import BaseQuantizeConfig, HQQLinear

def get_submodule_name(model, module_name):

    """
    module_name="blocks.11.mlp
    return model.blocks[11].mlp
    """

    if module_name == "":
        return model

    module = model
    for attr in module_name.split("."):
        if attr.isdigit():
            module = module[int(attr)]
        else:
            module = getattr(module, attr)

    return module

def replace_linear_with_hqq(
    model,                         # the original PyTorch model
    target_layers,                 # a list of module names, e.g., ["blocks.11.mlp.fc1", "blocks.11.mlp.fc2"]
    quant_config,
    compute_dtype=torch.float16,   # the dtype used during HQQ forward computation
    device="cuda",
    del_orig=True,                 # if True, delete the original Linear weight inside HQQLinear
    verbose=True,                  # if True, print replacement information
):

    replaced_layers = []

    # take "target_layers = ["blocks.11.mlp.fc1"]" for example
    for full_name in target_layers:
        if "." in full_name:
            parent_name, child_name = full_name.rsplit(".", 1)
            # parent_name = "blocks.11.mlp", child_name = "fc1"
        else:
            parent_name = ""
            child_name = full_name

        parent_module = get_submodule_name(model, parent_name)
        child_module = getattr(parent_module, child_name)
        # parent_module = model.blocks[11].mlp, child_module = model.blocks[11].mlp.fc1

        if not isinstance(child_module, torch.nn.Linear):
            raise TypeError(
                f"{full_name} is not torch.nn.Linear, "
                f"but got {type(child_module)}"
            )

        hqq_layer = HQQLinear(
            child_module,
            quant_config=quant_config,
            compute_dtype=compute_dtype,
            device=device,
            initialize=True,             # Use False to quantize later
            del_orig=del_orig,
        )

        setattr(parent_module, child_name, hqq_layer)

        replaced_layers.append(full_name)

        if verbose:
            print(f"[HQQ] Replaced {full_name}: Linear -> HQQLinear")

    return model


def get_all_linear_layer_names(
        model,
        exclude_names=None,
):
    """
    Get all torch.nn.Linear layer names in the model.
    Example return:
        [
            "blocks.0.attn.qkv",
            "blocks.0.attn.proj",
            "blocks.0.mlp.fc1",
            "blocks.0.mlp.fc2",
            ...
            "head"
        ]
    """

    if exclude_names is None:
        exclude_names = []

    linear_layer_names = []

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            if name in exclude_names:
                continue
            linear_layer_names.append(name)

    return linear_layer_names


def replace_all_linear_with_hqq(
    model,
    quant_config,
    compute_dtype=torch.float16,
    device="cuda",
    del_orig=True,
    verbose=True,
    exclude_names=None,
):
    """
    Replace all torch.nn.Linear layers in the model with HQQLinear.
    """

    target_layers = get_all_linear_layer_names(model, exclude_names=exclude_names)

    if verbose:
        print(f"[HQQ] Found {len(target_layers)} Linear layers.")
        for name in target_layers:
            print(f"  - {name}")

    model = replace_linear_with_hqq(
        model=model,
        target_layers=target_layers,
        quant_config=quant_config,
        compute_dtype=compute_dtype,
        device=device,
        del_orig=del_orig,
        verbose=verbose,
    )

    return model

def replace_all_linear_with_hqq_safe(
    model,
    nbits,
    group_size,
    compute_dtype=torch.float16,
    device="cuda",
    del_orig=True,
    verbose=True,
    exclude_names=None,
):
    """
    Same as replace_all_linear_with_hqq but:
    1. Excludes layers by substring match (e.g. "dt_proj" matches "blocks.0.spa_block.self_attention.dt_proj")
    2. Falls back to group_size=None for layers whose weight.numel() is not divisible by group_size
    """
    if exclude_names is None:
        exclude_names = []

    # get ALL linear layer names, no filtering yet
    all_linear_names = get_all_linear_layer_names(model, exclude_names=None)

    skipped = []
    replaced = []

    for full_name in all_linear_names:

        # substring match exclude
        if any(ex in full_name for ex in exclude_names):
            print(f"[HQQ] Skipping {full_name} (excluded)")
            skipped.append(full_name)
            continue

        if "." in full_name:
            parent_name, child_name = full_name.rsplit(".", 1)
        else:
            parent_name, child_name = "", full_name

        parent_module = get_submodule_name(model, parent_name)
        child_module  = getattr(parent_module, child_name)

        # safe group_size check
        # total_elements = child_module.weight.numel()
        # if group_size is not None and total_elements % group_size != 0:
        #     print(f"[HQQ] {full_name}: {child_module.weight.shape} not divisible by group_size={group_size}, falling back to group_size=None")
        #     effective_gs = None
        # else:
        effective_gs = group_size

        quant_config = BaseQuantizeConfig(nbits=nbits, group_size=effective_gs)

        hqq_layer = HQQLinear(
            child_module,
            quant_config=quant_config,
            compute_dtype=compute_dtype,
            device=device,
            initialize=True,
            del_orig=del_orig,
        )
        setattr(parent_module, child_name, hqq_layer)
        replaced.append(full_name)

        if verbose:
            print(f"[HQQ] Replaced {full_name} | nbits={nbits} | group_size={effective_gs}")

    print(f"\n[HQQ] Done. Replaced: {len(replaced)} | Skipped: {len(skipped)}")
    return model