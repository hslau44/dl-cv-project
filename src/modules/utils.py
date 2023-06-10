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


class DataFrameDatasetETL:
    
    def __init__(self,dataset):
        self.dataset = dataset
        self.target_dir = None
    
    def __call__(self,process_idx=None):
        p_idxs, n_idxs = [], []
        if not os.path.exists(self.target_dir):
            raise ValueError(f"Target directory is not a proper format or does not exist: {self.target_dir}")
        if process_idx is None:
            process_idx = range(len(self.dataset))
        print(f"Start Process, Total length: {len(process_idx)}")
        for i in process_idx:
            try:
                rtn_res = self.process_item(index=i)
                p_idxs.append(i)
                print(f"Complete file idx: {i} ----- {rtn_res}")
            except Exception as e:
                n_idxs.append(i)
                print(f"Imcomplete file idx: {i} ----- {e}")
        self.save_metadata(indexs=p_idxs)
        print("ETL completed")
        return n_idxs
    
    def process_item(self,index):
        item = self.get_item(index=index)
        filepath = self.get_filepath(index=index)
        rtn_res = self.save_item(item,filepath)
        return rtn_res
        
    def get_item(self,index):
        raise NotImplementedError
        
    def get_filepath(self,index):
        raise NotImplementedError
        
    def save_item(self,item,filepath):
        raise NotImplementedError
        
    def save_metadata(self,indexs):
        raise NotImplementedError
        
    def set_target_dir(self,target_dir):
        self.target_dir = target_dir
