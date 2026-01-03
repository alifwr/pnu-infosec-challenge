import torch


def inspect_model(model):
    """
    Prints the layers of the model to help identifying components.
    """
    print(f"\n{'Idx':<5} {'Type':<25} {'Params':<15} {'Arguments'}")
    print("-" * 80)

    # Access the internal nn.Sequential model
    # YOLO.model -> DetectionModel
    # DetectionModel.model -> nn.Sequential
    for i, m in enumerate(model.model.model):
        t = m.__class__.__name__
        p = sum(x.numel() for x in m.parameters())
        # Try to get some module args if possible, though exact args are hard to reconstruct from module
        # We can just print the type and param count
        print(f"{i:<5} {t:<25} {p:<15}")
    print("-" * 80 + "\n")


def load_specific_weights(model, weights_path, target_layers=None):
    """
    Loads weights from a pretrained model only for the specified layer indices.

    Args:
        model: The target YOLO model.
        weights_path: Path to the pretrained .pt file.
        target_layers: List of integer indices corresponding to the layers to load.
                       If None, loads nothing (or all? Safer to default to explicit).
    """
    if target_layers is None:
        print("No target layers specified, skipping weight loading.")
        return model

    print(f"Loading weights from {weights_path} for layers: {target_layers}")

    # Load the checkpoint
    # weights_only=False is required for full model checkpoints containing custom classes
    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)

    # Extract state dictionary
    if "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt

    if hasattr(state_dict, "state_dict"):
        state_dict = state_dict.state_dict()

    # Prepare new state dict
    filtered_state_dict = {}
    loaded_keys = []

    for key, value in state_dict.items():
        # structure is usually 'model.X.layer_name'
        parts = key.split(".")
        if len(parts) > 1 and parts[0] == "model" and parts[1].isdigit():
            layer_idx = int(parts[1])

            if layer_idx in target_layers:
                filtered_state_dict[key] = value
                loaded_keys.append(key)

    # Load into the current model
    model_wrapper = model.model

    # load_state_dict with strict=False
    missing, unexpected = model_wrapper.load_state_dict(
        filtered_state_dict, strict=False
    )

    print(f"Loaded {len(loaded_keys)} keys for {len(target_layers)} specified layers.")
    print(f"Total missing keys: {len(missing)}")

    return model
