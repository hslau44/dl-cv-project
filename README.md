# dl-cv-project (example: Kaggle WCE Curated Colon Disease Dataset)

This repository serves as a developing template for Deep Learning Project, the serving example project 'Image Classification of Colon Disease from Wireless Capsule Endoscopy', as the name suggestion, to train and deploy deep learning model to classify the type of colon disease of the image captured by wireless capsule endoscopy. The repository are designed to be task agnostic except the `src.wce`. 

The dataset used in this project is from the [Kaggle WCE Curated Colon Disease Dataset](https://www.kaggle.com/datasets/francismon/curated-colon-dataset-for-deep-learning?select=train), it is a curated data processed from the following sources:  
```
KVASIR Dataset (https://dl.acm.org/doi/abs/10.1145/3083187.3083212)
K. Pogorelov et al., "KVASIR", Proceedings of the 8th ACM on Multimedia Systems Conference, 2017. DOI: 10.1145/3083187.3083212.

ETIS-Larib-Polyp DB Dataset (https://link.springer.com/article/10.1007/s11548-013-0926-3/)
J. Silva, A. Histace, O. Romain, X. Dray and B. Granado, "Toward embedded detection of polyps in WCE images for early diagnosis of colorectal cancer", International Journal of Computer Assisted Radiology and Surgery, vol. 9, no. 2, pp. 283-293, 2013. DOI:10.1007/s11548-013-0926-3.

F. J. P. Montalbo, "Diagnosing Gastrointestinal Diseases from Endoscopy Images through a Multi-Fused CNN with Auxiliary Layers, Alpha Dropouts, and a Fusion Residual Block," Biomedical Signal Processing and Control (BSPC), vol. 76, July, 2022, doi: 10.1016/j.bspc.2022.103683
```

## Setup

##### Code #####
1. run command  `pip install -r requirements.txt`

##### Data #####
1. Download the dataset from this [link](https://www.kaggle.com/datasets/francismon/curated-colon-dataset-for-deep-learning?select=train) and extract the file to the `data/`



## Overview (updating)


### strucutre

```
├── data/
├── notebooks/
├── outputs/
├── src/
│   ├── modules/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── models.py
│   │   ├── processor.py
│   │   └── utils.py
│   │
│   ├── __init__.py
│   ├── __main__.py
│   ├── __config__.py
│   ├── serve.py
│   ├── train.py
│   ├── utils.py
│   └── wce.py
├── tests/
├── README.md
└── requirements.txt
```


## Run Example: 

##### WCE Training with configuration #####

Please look at `notebooks/WCE_Training_with_configuration.ipynb`  
