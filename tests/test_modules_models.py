import torch
from src.modules.models import TransferModel

def test_TransferModel():

    torch.manual_seed(0)

    bs = 4
    num_label = 8

    input_values = torch.rand((bs,3,224,224))

    model = TransferModel(
        base_model='alexnet',
        proj_features=[128,num_label],
        proj_biases=[True,False],
        proj_activations=['relu',None]
    )

    assert tuple(model(input_values).shape) == (bs,num_label)
