import torch
import torch.nn as nn

def module():
    # concept
    """
    torch.nn: nn.Parameter + nn.Module + nn.functional + nn.init
    nn.Module: nn.Parameter + nn.Module + buffers + *_hooks
    """

    # forward
    """
    def __call__(self, *input, **kwargs):
        result = self.forward(*input, **kwargs)
        for hook in self._forward_hooks.values():
            hook_result = hook(self, input, result)
        ...
        return result
    """

    # container: nn.Sequential, nn.MoudleList, nn.ModuleDict

    # layers: nn.Conv2d, nn.ConvTranspose2d, nn.MaxPool2d, nn.AvgPool2d, nn.MaxUnpool2d
    """
    nn.Conv2d(
        in_channels, 
        out_channels, 
        kernel_size, 
        stride=1, 
        padding=0, 
        dilation=1,  # dilation conv, used in segmentation
        groups=1,  # group conv, used in shufflenet, mobilenet
        bias=True,
        padding_mode='zeros'
    )
    """

    """
    nn.ConvTranspose2d(
        in_channels, 
        out_channels, 
        kernel_size, 
        stride=1, 
        padding=0, 
        dilation=1,  
        groups=1,  
        bias=True,
        padding_mode='zeros'
    )
    """

    """
    nn.MaxPool2d(
        kernel_size, 
        stride=None, 
        padding=0, 
        dilation=1,
        return_indices=False,
        ceil_mode=False
    )
    """

    """
    nn.AvgPool2d(
        kernel_size, 
        stride=None, 
        padding=0, 
        ceil_mode=False,
        count_include_pad=True, 
        divisor_override=None
    )
    """

    """
    nn.MaxUnpool2d(
        kernel_size,
        stride=None,
        padding=0
    )
    # forward(self, input, indices, output_size=None)
    """


def save_and_load():
    # save object: model structure + parameters
    """
    torch.save(model, path)
    model = torch.load(path)
    """

    # save parameters:
    """
    net = MLP()
    torch.save(net.state_dict(), path)

    net2 = MLP()
    net2.load_state_dict(torch.load(path))
    """


def init():
    pass
