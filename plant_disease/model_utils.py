from __future__ import annotations


def create_model(num_classes: int, pretrained: bool = False):
    """Create EfficientNet-B0 with an output layer matching dataset classes."""
    import torch.nn as nn
    from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

    weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model
