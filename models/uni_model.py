import torch
import timm


def load_uni_model(config):
    """
    Load the UNI foundation model.

    Args:
        config: Configuration dictionary

    Returns:
        UNI model in eval mode
    """
    uni_config = config["model"]["uni"]

    timm_kwargs = {
        "img_size": uni_config["img_size"],
        "patch_size": uni_config["patch_size"],
        "depth": uni_config["depth"],
        "num_heads": uni_config["num_heads"],
        "init_values": uni_config["init_values"],
        "embed_dim": uni_config["embed_dim"],
        "mlp_ratio": uni_config["mlp_ratio"],
        "num_classes": uni_config["num_classes"],
        "no_embed_class": uni_config["no_embed_class"],
        "mlp_layer": timm.layers.SwiGLUPacked,
        "act_layer": torch.nn.SiLU,
        "reg_tokens": 8,
        "dynamic_img_size": uni_config["dynamic_img_size"],
    }

    model = timm.create_model(uni_config["model_name"], pretrained=True, **timm_kwargs)

    model.eval()
    return model
