from torch import nn
import torchvision.models as preset_model


ACTIVATION_DICT = {
    'relu': nn.ReLU,
    'none': nn.Identity,
    None: nn.Identity,
}

LINEAR_KEYS = {
    'out_features':'out_features',
    'bias':'bias',
    'activation':'activation',
}

PRETRAIN_MDL_CONFIG = {
    'alexnet': {
        'model_init': preset_model.alexnet, 
        'weights': preset_model.AlexNet_Weights.IMAGENET1K_V1,
        'out_features': 1000,
    },
}

class BaseModel(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self,input_values,**kwargs):
        raise NotImplementedError
        
    def _get_output_dim(self):
        raise NotImplementedError


class Projection(BaseModel):

    def __init__(self,in_features,out_features,biases,activations):
        super().__init__()
        self.linears = nn.ModuleList()
        in_feat = in_features
        for out_feat, bs, actv in zip(out_features,biases,activations):
            self.linears.append(nn.Linear(
                in_features=in_feat,
                out_features=out_feat,
                bias=bs)
            )
            self.linears.append(ACTIVATION_DICT[actv]())
            in_feat = out_feat
        self._output_dim = (in_feat,)

    def forward(self, input_values, **kwargs):
        x = input_values
        for layer in self.linears:
            x = layer(x)
        return x
    
    def _get_output_dim(self) -> tuple:
        return self._output_dim
    

class TransferModel(BaseModel):

    def __init__(self,base_model, proj_features, proj_biases, proj_activations):
        super().__init__()
        config = PRETRAIN_MDL_CONFIG[base_model]
        self.base = config['model_init'](weights=config['weights'])
        base_out_feat = config['out_features']
        self.projection = Projection(
            in_features=base_out_feat,
            out_features=proj_features,
            biases=proj_biases,
            activations=proj_activations,
        )
    
    def forward(self, input_values, **kwargs):
        input_values = self.base(input_values)
        return self.projection(input_values)
    
    def _get_output_dim(self):
        return self.projection._get_output_dim()
