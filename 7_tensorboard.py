import torch
import torch.nn as nn

import numpy as np
from torch.utils.tensorboard import SummaryWriter

def tensorboard():
    writer = SummaryWriter(comment='test_tensorboard')
    for x in range(100):
        writer.add_scalar('y=2x', x*2, x)
        writer.add_scalar('y=pow(2, x)', 2**x, x)
        writer.add_scalars('data/scalar_group', 
            {"xsinx": x * np.sin(x),
             "xcosx": x * np.cos(x), 
             "arctanx": np.arctan(x)}, x
        )
    writer.close()


if __name__ == '__main__':
    tensorboard()