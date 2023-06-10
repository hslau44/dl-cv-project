import os
import pandas as pd
from torch.utils.data import DataLoader
from .config import DATA_ARG_KEYS, SPLIT_SET_KEYS
from .modules.processors import Pipeline
from .modules.utils import DataFrameDataset, get_filepaths
from .modules.models import TransferModel, ACTIVATION_DICT


METADATA_COLNAMES = {
    'split_set':'split_set',
    'label':'label',
    'file_id':'file_id',
    'filepath':'filepath',
    'pt_file_path':'pt_file_path',
    'local_filepath':'local_filepath',
}

MASK_GREEN = {
    'target_rgb':(18,166,136), 
    'tol_rgb':25, 
    'mask_rgb':(0,0,0), 
    'mask_range':10,
}

MASK_TEXT = {
    'target_rgb':(0,0,0), 
    'tol_rgb':10, 
    'mask_rgb':(0,0,0), 
    'mask_range':10,
}

WCE_SPLIT_SET_KEYS = {
    SPLIT_SET_KEYS['train']: 'train',
    SPLIT_SET_KEYS['val']: 'val',
    SPLIT_SET_KEYS['test']: 'test',
    SPLIT_SET_KEYS['all']:'all',
}


DATASET_ITEM_KEYS = DATA_ARG_KEYS

WCE_LABEL_KEYS = {
    'normal':0,
    'ulcer':1,
    'polyps':2,
    'esophagitis':3,
    'ulcerative_colitis':1
}


WCE_STANDARD_PROCESSOR_CONFIG = {
    'ImageReader': None,
    'ColorMasker': MASK_GREEN,
    'ColorMasker': MASK_TEXT,
    'ToTensor':  None,
    'Resize': {
        'size':(224,224),
        'antialias':True,
    },
}


WCE_NPY_PROCESSOR_CONFIG = {
    'NpyReader': None,
    'ToTensor':  None,
}


WCE_LABEL_PROCESSOR_CONFIG = {
    'LabelToTensor': {
        'mapping': WCE_LABEL_KEYS,
    },
}

WCE_MODEL_CONFIG = {
    'num_label': 4,
    'last_bias': True, 
    'last_activation': 'none',
}

WCE_STANDARD_MODEL_CONFIG = {
    'base_model': 'alexnet', 
    'num_proj': 2, 
    'hid_feature': 128, 
    'hid_bias': False,
    'hid_activation': 'relu',
}


WCEStandardProcessor = lambda: Pipeline(WCE_STANDARD_PROCESSOR_CONFIG)


WCENpyProcessor = lambda: Pipeline(WCE_NPY_PROCESSOR_CONFIG)


WCELabelProcessor = lambda: Pipeline(WCE_LABEL_PROCESSOR_CONFIG)


def get_metadata(paths, rtn_pd=False):
    df = []
    for path in paths:
        if path.split('.')[-1] == 'jpg':
            dic = {}
            info = path.split(os.path.sep)[-1].split('_')
            dic[METADATA_COLNAMES['split_set']] = info[0]
            dic[METADATA_COLNAMES['label']] = info[1]
            dic[METADATA_COLNAMES['file_id']] = int(info[-1].strip(' (')[:-5])
            dic[METADATA_COLNAMES['filepath']] = os.path.join(*path.split('/')[-3:])
            # dic[METADATA_COLNAMES['local_filepath']] = path
            df.append(dic)
    df = pd.DataFrame(df) if rtn_pd else df
    return df


class WCEStandardDataset(DataFrameDataset):

    def __init__(self,dataset_dir,split_set):

        if not os.path.exists(dataset_dir):
            raise FileExistsError(f"dataset_dir '{dataset_dir}' does not exist")

        self.dataset_dir = dataset_dir
        self.split_set = split_set

        metadata = get_metadata(get_filepaths(dataset_dir=dataset_dir),True)

        if METADATA_COLNAMES['split_set'] not in metadata.columns:
            raise ValueError(f"Column 'split_set' does not exist")

        if split_set == WCE_SPLIT_SET_KEYS['all']:
            metadata = metadata
        elif split_set in WCE_SPLIT_SET_KEYS.keys():
            metadata = metadata[metadata[METADATA_COLNAMES['split_set']] == split_set].reset_index(drop=True)
        else:
            raise ValueError(f"'split_set' is not in {WCE_SPLIT_SET_KEYS.keys()}")
        
        metadata[METADATA_COLNAMES['local_filepath']] = metadata[METADATA_COLNAMES['filepath']].apply(
            self._get_full_filepath
        )
        
        transfrom_map = {
            METADATA_COLNAMES['local_filepath']: WCEStandardProcessor(),
            METADATA_COLNAMES['label']: WCELabelProcessor()
        }

        output_keys = [
            DATA_ARG_KEYS['input_values'],
            DATA_ARG_KEYS['label']
        ]

        super().__init__(metadata=metadata,transfrom=transfrom_map,output_key=output_keys)

    def __len__(self):
        return len(self.metadata)
    
    def _get_full_filepath(self,filepath):
        return os.path.join(self.dataset_dir,filepath)


class WCEStandardNpyDataset(DataFrameDataset):

    def __init__(self,dataset_dir,split_set):

        if not os.path.exists(dataset_dir):
            raise FileExistsError(f"dataset_dir '{dataset_dir}' does not exist")

        self.dataset_dir = dataset_dir
        self.split_set = split_set

        metadata = pd.read_csv(os.path.join(dataset_dir,'metadata.csv'))

        if METADATA_COLNAMES['split_set'] not in metadata.columns:
            raise ValueError(f"Column 'split_set' does not exist")

        if split_set == WCE_SPLIT_SET_KEYS['all']:
            metadata = metadata
        elif split_set in WCE_SPLIT_SET_KEYS.keys():
            metadata = metadata[metadata[METADATA_COLNAMES['split_set']] == split_set].reset_index(drop=True)
        else:
            raise ValueError(f"'split_set' is not in {WCE_SPLIT_SET_KEYS.keys()}")
        
        metadata[METADATA_COLNAMES['local_filepath']] = metadata[METADATA_COLNAMES['filepath']].apply(
            self._get_full_filepath
        )
        
        transfrom_map = {
            METADATA_COLNAMES['local_filepath']: WCENpyProcessor(),
            METADATA_COLNAMES['label']: WCELabelProcessor()
        }

        output_keys = [
            DATA_ARG_KEYS['input_values'],
            DATA_ARG_KEYS['label']
        ]

        super().__init__(metadata=metadata,transfrom=transfrom_map,output_key=output_keys)

    def __len__(self):
        return len(self.metadata)
    
    def _get_full_filepath(self,filepath):
        return os.path.join(self.dataset_dir,filepath)

    
class WCEStandardDataloader(DataLoader):
    
    def __init__(self,dataset_dir,split_set,**kwargs):
        dataset = WCEStandardDataset(
            dataset_dir=dataset_dir,
            split_set=split_set
        )
        super().__init__(dataset=dataset,**kwargs)


class WCEStandardNpyDataloader(DataLoader):
    
    def __init__(self,dataset_dir,split_set,**kwargs):
        dataset = WCEStandardNpyDataset(
            dataset_dir=dataset_dir,
            split_set=split_set
        )
        super().__init__(dataset=dataset,**kwargs)


class WCETransferModel(TransferModel):

    def __init__(self, base_model, num_proj, hid_feature, hid_bias, hid_activation, **kwargs):

        proj_features = []
        proj_biases = []
        proj_activations = []

        for i in range(num_proj-1):
            proj_features.append(hid_feature)
            proj_biases.append(hid_bias)
            proj_activations.append(hid_activation)
        
        proj_features.append(WCE_MODEL_CONFIG['num_label'])
        proj_biases.append(WCE_MODEL_CONFIG['last_bias'])
        proj_activations.append(WCE_MODEL_CONFIG['last_activation'])

        super().__init__(base_model, proj_features, proj_biases, proj_activations)
