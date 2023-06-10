import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
import lightning.pytorch.callbacks as C
import lightning.pytorch.loggers as L 
from .config import DATA_ARG_KEYS, SPLIT_SET_KEYS
from .utils import get_cls_arg_pair, get_cls_arg_pair_list, write_json


OPTIM_INIT_KEYS = {
    'adam': torch.optim.Adam
}

def process_score(metrics):
    return {k: float(v) for k,v in metrics.items()}


class EarlyStopping(C.early_stopping.EarlyStopping):
    pass

class CSVLogger(L.CSVLogger):
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
        self.lost_func = nn.CrossEntropyLoss()

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        input_values = batch[DATA_ARG_KEYS['input_values']]
        label = batch[DATA_ARG_KEYS['label']]
        logits =  self.model(input_values)
        loss = self.lost_func(logits, label)
        return loss
    
    def validation_step(self, batch, batch_idx):
        # training_step defines the train loop.
        input_values = batch[DATA_ARG_KEYS['input_values']]
        label = batch[DATA_ARG_KEYS['label']]
        logits =  self.model(input_values)
        loss = self.lost_func(logits, label)
        self.log("val_loss", loss)

    def configure_optimizers(self):
        optim_init = OPTIM_INIT_KEYS[self.optim_name]
        optimizer = optim_init(self.parameters(), lr=self.lr)
        return optimizer


def objective(args):

    score = {}
    write_json(args,os.path.join(args['default_root_dir'],'configs.json'))

    # init 
    dataloader_init, dataloader_args = get_cls_arg_pair(args.pop('dataloader'))
    model_init, model_args = get_cls_arg_pair(args.pop('model'))
    pl_module_init, pl_module_args = get_cls_arg_pair(args.pop('pl_module'))
    logger_init, logger_args = get_cls_arg_pair(args.pop('logger'))
    
    # data 
    train_dataloader = dataloader_init(
        split_set=SPLIT_SET_KEYS['train'],
        **dataloader_args
    )
    val_dataloader = dataloader_init(
        split_set=SPLIT_SET_KEYS['val'],
        **dataloader_args
    )

    # model
    model = model_init(**model_args)
    pl_module = pl_module_init(
        model=model,
        **pl_module_args
    )

    callbacks = [i(**a) for i, a in get_cls_arg_pair_list(args.pop('callbacks'))]
    logger = logger_init(**logger_args)
    trainer = pl.Trainer(callbacks=callbacks,logger=logger,**args)

    # train start 
    trainer.fit(pl_module,train_dataloader,val_dataloader)
    score = process_score(trainer.callback_metrics)
    
    write_json(score,os.path.join(args['default_root_dir'],'scores.json'))
    
    return score