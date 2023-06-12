from .train import objective


def get_example_configs():
    root_dir = "./outputs/example"
    objective_metric = 'val_loss'
    configs = {
        # init args pairs
        'dataloader': {
            'wce.WCEStandardNpyDataloader': {
                'dataset_dir': './data/std_npy',
                'batch_size': 32,
                'num_workers': 4,
                'drop_last': True,
        }},
        'model': {
            'wce.WCETransferModel':{
                'base_model': 'alexnet', 
                'num_proj': 2, 
                'hid_feature': 128, 
                'hid_bias': False,
                'hid_activation': 'relu',
        }},
        'pl_module': {
            'train.SupervisedClfModule': {
                'optim_name': 'adam',
                'lr': 1e-4,
        }},
        'logger': {
            'train.CSVLogger': {
                'save_dir': root_dir, 
                'name':"csv_log",
        }},
        'callbacks': {
            'list': { 
                1: {'train.EarlyStopping': {
                        'monitor':objective_metric, 
                        'mode':"min",
                    }
                }
            }
        },
        #### pl.Trainer args
        'default_root_dir': root_dir,
        'max_epochs':2,
        # 'accelerator':"gpu", 
        # 'devices':-1,
        # 'fast_dev_run':True,
        #### others
        'objective_metric': objective_metric
    }
    return configs


config = get_example_configs()
results = objective(config)
print(results)

print("*****Complete*****")