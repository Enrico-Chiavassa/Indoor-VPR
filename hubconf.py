dependencies = ['torch', 'torchvision']

import torch
from model.network import ModelSelecter

AVAILABLE_TRAINED_MODELS = {
    # backbone : list of available fc_output_dim, which is equivalent to descriptors dimensionality
    "eigenplaces":  {
      "ResNet50": [2048]
    },
    "salad": {
      "Dinov2": [8448]
    },
}

# For each combination 2 different models are available. They differ in the dataset used for the fine-tuning,
# the thresholds used to select soft and hard positives and in the way the images are handled during the training procedure.

AVAILABLE_VARIATIONS = {
  "eigenplaces_ResNet50_2048": ["GB1_BAI_5_10", "GB1_BAI_10_25_S"],
  "salad_Dinov2_8448": ["GB1_10_25", "HB1_GB1_2_5"],
}


def get_trained_model(method : str = "eigenplaces", backbone : str = "ResNet50", fc_output_dim : int = 2048, variation : int = 0) -> torch.nn.Module:
    """Return a model fine-tuned on indoor datasets.
    
    Args:
        method (str): which methods was used to firstly train the model.
        backbone (str): which torchvision backbone to use. Must be ViT or a ResNet.
        fc_output_dim (int): the output dimension of the last fc layer, equivalent to
            the descriptors dimension. Must be between 32 and 2048, depending on model's availability.
    
    Return:
        model (torch.nn.Module): a trained model.
    """
    print(f"Returning {method} model with backbone: {backbone} with features dimension {fc_output_dim}")
    if method not in AVAILABLE_TRAINED_MODELS:
      raise ValueError(f"Parameter `method` is set to {method} but it must be one of {list(AVAILABLE_TRAINED_MODELS.keys())}")
      
    if backbone not in AVAILABLE_TRAINED_MODELS[method]:
        raise ValueError(f"Parameter `backbone` is set to {backbone} but it must be one of {list(AVAILABLE_TRAINED_MODELS[method].keys())}")
      
    try:
        fc_output_dim = int(fc_output_dim)
    except:
        raise ValueError(f"Parameter `fc_output_dim` must be an integer, but it is set to {fc_output_dim}")
      
    if fc_output_dim not in AVAILABLE_TRAINED_MODELS[method][backbone]:
        raise ValueError(f"Parameter `fc_output_dim` is set to {fc_output_dim}, but for backbone {backbone} "
                         f"it must be one of {list(AVAILABLE_TRAINED_MODELS[backbone])}")

    if variation not in [0,1]:
      raise ValueError(f"Parameter `variation` is set to {variation}, but must be 0 or 1")

    file_name = f"{method}_{backbone}_{fc_output_dim}"
    var_name = AVAILABLE_VARIATIONS[file_name][variation]
    file_name += f"_{var_name}"

    fetched_model = ModelSelecter(backbone, fc_output_dim).return_model()
    fetched_model.load_state_dict(
        torch.hub.load_state_dict_from_url(
            f'https://github.com/Enrico-Chiavassa/Indoor-VPR/releases/download/v0.1.0/{file_name}.pth',
        map_location=torch.device('cpu'))
    )
    return fetched_model
