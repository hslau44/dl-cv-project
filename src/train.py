import torch
import torch.nn.functional as F
import lightning.pytorch as pl
import lightning.pytorch.callbacks as C
from .config import DATA_ARG_KEYS, SPLIT_SET_KEYS
from .utils import get_cls_arg_pair, get_cls_arg_pair_list


OPTIM_INIT_KEYS = {
    'adam': torch.optim.Adam
}


class EarlyStopping(C.early_stopping.EarlyStopping):
    pass


class BasePLModule(pl.LightningModule):

    def __init__(self):
        super().__init__()

    def training_step(self, batch, batch_idx):
        raise NotImplementedError
    
    def validation_step(self, batch, batch_idx):
        raise NotImplementedError
    
    def configure_optimizers(self):
        raise NotImplementedError


class SupervisedClfModule(BasePLModule):
    def __init__(self, model, optim_name, lr):
        super().__init__()
        self.model = model
        self.optim_name = optim_name
        self.lr = lr

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        input_values = batch[DATA_ARG_KEYS['input_values']]
        label = batch[DATA_ARG_KEYS['label']]
        logits =  self.model(input_values)
        loss = F.nll_loss(logits, label)
        return loss
    
    def validation_step(self, batch, batch_idx):
        # training_step defines the train loop.
        input_values = batch[DATA_ARG_KEYS['input_values']]
        label = batch[DATA_ARG_KEYS['label']]
        logits =  self.model(input_values)
        loss = F.nll_loss(logits, label)
        self.log("val_loss", loss)

    def configure_optimizers(self):
        optim_init = OPTIM_INIT_KEYS[self.optim_name]
        optimizer = optim_init(self.parameters(), lr=self.lr)
        return optimizer
