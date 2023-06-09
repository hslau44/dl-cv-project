import pytest
import torch
from src.modules.models import TransferModel
from .utils import get_testdata


@pytest.mark.parametrize("batch_size,num_label", get_testdata(6,['int','int'],[[2,6],[2,6]]))
def test_TransferModel(batch_size,num_label):

    torch.manual_seed(0)

    bs = batch_size # 4
    # num_label = 8

    input_values = torch.rand((bs,3,224,224))

    model = TransferModel(
        base_model='alexnet',
        proj_features=[128,num_label],
        proj_biases=[True,False],
        proj_activations=['relu',None]
    )

    assert tuple(model(input_values).shape) == (bs,num_label)
