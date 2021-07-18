import torch
import torch.nn as nn


# Sampler: generate index

# Dataset: getitem of index -> (img, label)
"""
class Dataset(object):
    def __getitem__(self, index):
        raise NotImplementError
    
    def __add__(self, other):
        return ConcatDataset([self, other])
"""
# collate_fn: generate batch
# DataLoader
"""
DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    sampler=None,
    batch_sampler=None,
    num_workers=0,
    collate_fn=None,
    pin_memory=False,
    drop_last=False,
    timeout=0,
    worker_init_fn=None,
    multiprocessing_context=None
    )
"""
