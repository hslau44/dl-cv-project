import os
import sys
import importlib
from .modules.utils import example_func


def get_cls_in_src(name):

    path = name.split('.')

    if path[0] == 'src':
        ls = []
    elif 'src' not in path:
        ls = ['src']
    else:
        raise ValueError()

    ls += path[:-1]

    package_name = '.'.join(ls) 

    package = importlib.import_module(package_name)

    _cls = getattr(package, path[-1])

    return _cls


def get_cls_arg_pair(dic):

    name = list(dic.keys())[0]

    _cls = get_cls_in_src(name)

    _args = dic[name]

    return _cls, _args


def get_cls_arg_pair_list(dic):
    
    name = list(dic.keys())[0]

    if name == 'list':

        ls  = []

        for idx in dic['list'].keys():

            item_dic = dic['list'][idx]

            pair = get_cls_arg_pair(item_dic)

            ls.append(pair)
        
        return ls
    
    else:

        _cls, _args = get_cls_arg_pair(dic)

        return _cls, _args
