import sys
from typing import Union
import numpy as np
import scipy
import cv2
import torch
import torchvision.transforms as T
from .config import *


class BaseProcessor(object):

    def __init__(self,**kwargs):
        pass
    
    def __call__(self,data):
        return self.process_data(data)
    
    def process_data(self,data):
        raise NotImplementedError

    
class BatchProcessor(BaseProcessor):
    
    def __call__(self,data):
        if isinstance(data,list):
            return [self.process_data(d) for d in data]
        else:
            return self.process_data(data)
    

class ComposedProcessor(BaseProcessor):
    
    def __init__(self,*processor):
        self.processor = processor
        
    def process_data(self,data):
        for p in self.processor:
            data = p(data)
        return data
    
    def __repr__(self) -> str:
        string = f"{self.__class__.__name__}(\n"
        for p in self.processor:
            string += '    ' + p.__class__.__name__ + '\n'
        string += ')'
        return string
        

class ImageReader(BaseProcessor):
    """
    Read image in RGB
    """
    def __init__(self,**kwargs):
        pass
    
    def process_data(self,data):
        im = cv2.imread(data)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        return im


class ColorMasker(BaseProcessor):
    """
    Mask region in RGB
    """
    def __init__(
        self,
        target_rgb: tuple, 
        tol_rgb: Union[tuple, int], 
        mask_rgb: tuple, 
        mask_range: Union[tuple, int],
        **kwargs
    ):
        
        if isinstance(tol_rgb,int):
            tol_rgb = (tol_rgb, tol_rgb, tol_rgb)
        if isinstance(mask_range,int):
            mask_range = (mask_range, mask_range)

        if len(target_rgb) != 3:
            raise ValueError("'target_rgb' dim must be equal to 3")
        if len(tol_rgb) != 3:
            raise ValueError("'tol_rgb' dim must be equal to 3")
        if len(mask_rgb) != 3:
            raise ValueError("'mask_rgb' dim must be equal to 3")
        if len(mask_range) != 2:
            raise ValueError("'mask_range' dim must be equal to 2")
            
        
        self.low  = np.array([max(0,target_rgb[i] - tol_rgb[i]) for i in range(3)])
        self.high = np.array([min(255,target_rgb[i] + tol_rgb[i]) for i in range(3)])
        self.kernel = np.ones(mask_range)
        self.mask_rgb = mask_rgb
        
    def process_data(self,data):
        mask = cv2.inRange(data,self.low,self.high)
        mask = self._get_convoluted_mask(mask)
        data[mask>0] = self.mask_rgb
        return data
    
    def _get_convoluted_mask(self,mask):
        new_mask = scipy.signal.convolve2d(mask,self.kernel,mode='same').astype(bool).astype(int)
        return new_mask
    

class LabelToTensor(BaseProcessor):
    
    def __init__(self,mapping):
        self._map = mapping
        self.to_tensor = T.ToTensor()
    
    def process_data(self,data):
        lb = self._map[data]
        return torch.tensor(lb,dtype=torch.int8).long()
    

class ToTensor(T.ToTensor):
    pass

class Resize(T.Resize):
    pass


class Pipeline(ComposedProcessor):

    def __init__(self,config):
        processors = []
        
        for cls_name, cls_config in config.items():
            _cls = getattr(sys.modules[__name__],cls_name)
            if cls_config is None:
                processors.append(_cls())
            else:
                processors.append(_cls(**cls_config))
        super().__init__(*processors)
