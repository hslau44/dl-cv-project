import os
from src.wce import (
    WCEStandardDataloader,
    WCEStandardNpyDataloader,
    WCEStandardDataset, 
    WCEStandardNpyDataset,
    WCETransferModel,
    get_metadata, 
    get_filepaths, 
    DATASET_ITEM_KEYS,
    SPLIT_SET_KEYS,
    WCE_MODEL_CONFIG,
    WCE_STANDARD_MODEL_CONFIG,
)

DATASET_DIR = {
    'standard': '../data/original',
    'std_npy': '../data/std_npy'
}


def test_WceStandardDataset():
    dataset_dir = DATASET_DIR['standard']

    _set = SPLIT_SET_KEYS['all']
    dataset = WCEStandardDataset(dataset_dir=dataset_dir,split_set=_set)

    img_shape = (3, 224, 224)

    test_infos = {
        3: {'shape':img_shape, 'label': 0},
        300: {'shape':img_shape, 'label': 1},
        3000: {'shape':img_shape, 'label': 2},
    }

    for i,info in test_infos.items():

        k = dataset[i]
        
        s = k[DATASET_ITEM_KEYS['input_values']].shape
        l = k[DATASET_ITEM_KEYS['label']]

        assert info['shape'] == tuple(s)
        assert info['label'] == int(l)
        
        
def test_WceStandardNpyDataset():
    dataset_dir = DATASET_DIR['std_npy']

    _set = SPLIT_SET_KEYS['all']
    dataset = WCEStandardNpyDataset(dataset_dir=dataset_dir,split_set=_set)

    img_shape = (3, 224, 224)

    test_infos = {
        3: {'shape':img_shape, 'label': 0},
        300: {'shape':img_shape, 'label': 1},
        3000: {'shape':img_shape, 'label': 2},
    }

    for i,info in test_infos.items():

        k = dataset[i]
        
        s = k[DATASET_ITEM_KEYS['input_values']].shape
        l = k[DATASET_ITEM_KEYS['label']]

        assert info['shape'] == tuple(s)
        assert info['label'] == int(l)


def test_WceStandardDataloader():

    batch_size = 4
    img_shape = (3, 224, 224)
    

    args = {}
    args.update(
        dataset_dir=DATASET_DIR['standard'],
        split_set=SPLIT_SET_KEYS['val'],
        batch_size=batch_size,
        shuffle=True,
    )
    batch_shape = (batch_size,*img_shape)

    dl = WCEStandardDataloader(**args)
    batch = next(iter(dl))

    b_s = batch[DATASET_ITEM_KEYS['input_values']].shape
    b_ls = batch[DATASET_ITEM_KEYS['label']].shape

    assert tuple(b_s) == batch_shape
    assert len(b_ls) == 1
    assert b_ls[0] == batch_size
    
    
def test_WceStandardNpyDataloader():

    batch_size = 4
    img_shape = (3, 224, 224)
    

    args = {}
    args.update(
        dataset_dir=DATASET_DIR['std_npy'],
        split_set=SPLIT_SET_KEYS['val'],
        batch_size=batch_size,
        shuffle=True,
    )
    batch_shape = (batch_size,*img_shape)

    dl = WCEStandardNpyDataloader(**args)
    batch = next(iter(dl))

    b_s = batch[DATASET_ITEM_KEYS['input_values']].shape
    b_ls = batch[DATASET_ITEM_KEYS['label']].shape

    assert tuple(b_s) == batch_shape
    assert len(b_ls) == 1
    assert b_ls[0] == batch_size


def test_WceTransferModel():

    bs = 4
    num_label = WCE_MODEL_CONFIG['num_label']

    args = {}
    args.update(
        dataset_dir=DATASET_DIR['standard'],
        split_set=SPLIT_SET_KEYS['val'],
        batch_size=bs,
        shuffle=True,
    )
    dl = WCEStandardDataloader(**args)

    batch = next(iter(dl))

    input_values = batch[DATASET_ITEM_KEYS['input_values']]

    model = WCETransferModel(**WCE_STANDARD_MODEL_CONFIG)

    assert tuple(model(input_values).shape) == (bs,num_label)
    assert model._get_output_dim() == (num_label,)
    

if __name__ == "__main__":
    pass