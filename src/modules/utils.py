import os
import pandas as pd
from typing import Callable, Dict, List
from torch.utils.data import Dataset
from .config import *


def example_func():
    """Example function takes no arguments and return True"""
    return True


def get_filepaths(dataset_dir):
    """get all filepath in dir"""
    paths = []
    for dirname, _, filenames in os.walk(dataset_dir):
        for filename in filenames:
            path = os.path.join(dirname, filename)
            paths.append(path)
    # sort all paths
    paths.sort()
    return paths


class DataFrameDataset(Dataset):
    """
    Data Generater extracts value from specific 
    index and column of the metadata, transfrom
    the value and output the value by its given 
    output key 
    """
    def __init__(self, metadata: pd.DataFrame, transfrom: Dict[str,Callable], output_key: List):

        for k in transfrom.keys():
            if k not in metadata.columns:
                raise ValueError(f"transform key: {k} is not an available column name in metadata")

        if len(output_key) != len(transfrom.keys()):
            raise ValueError("Length of 'output_key' and 'transfrom' are not identical.")
        
        self.metadata = metadata
        self.transfrom = transfrom
        self.output_key = output_key

    def __getitem__(self,idx):
        sample = {}
        info = dict(self.metadata.iloc[idx,:])
        for (key1, piepline), key2 in zip(self.transfrom.items(),self.output_key):
            data = info[key1]
            sample[key2] = piepline(data)
        return sample 
    
    def __len__(self):
        return len(self.metadata)
