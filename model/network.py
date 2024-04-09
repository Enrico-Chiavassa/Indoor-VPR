import torch
import torchvision
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class ModelSelecter():
    def __init__(self, backbone: str, fc_output_dim: int):
        self.model = get_model(backbone=backbone, fc_output_dim=fc_output_dim)
    def return_model(self):
        return self.model
        
def get_model(backbone: str, fc_output_dim: 2048):
    if backbone == "ResNet50":
        return GeoLocalizationNet(backbone=backbone, fc_output_dim=fc_output_dim)
    elif backbone == "Dinov2":
        return GeoLocalizationViT()
    return model

class GeoLocalizationNet(torch.nn.Module):
    """The used networks are composed of a backbone and an aggregation layer.
    """
    def __init__(self, backbone: str, fc_output_dim: int):
        super().__init__()
        self.model = torch.hub.load("gmberton/eigenplaces", "get_trained_model", backbone=backbone, fc_output_dim=fc_output_dim)
    def forward(self, x):
        x = self.model(x)
        return x

class GeoLocalizationViT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.hub.load("serizba/salad", "dinov2_salad", backbone="dinov2_vitb14")
    def forward(self, images):
        b, c, h, w = images.shape
        # DINO wants height and width as multiple of 14, therefore resize them
        # to the nearest multiple of 14
        h = round(h / 14) * 14
        w = round(w / 14) * 14
        images = torchvision.transforms.functional.resize(images, [h, w], antialias=True)
        return self.model(images)
