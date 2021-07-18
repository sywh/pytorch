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
    pass


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
    pass


def init():
    pass


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 2, 3)
        self.pool1 = nn.MaxPool2d(2)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        return x

def hook():
    # torch.Tensor.register_hook
    # hook(grad) -> Tensor or None
    w = torch.tensor([1.], requires_grad=True)
    x = torch.tensor([2.], requires_grad=True)
    a = torch.add(w, x)
    b = torch.add(w, 1)
    y = torch.mul(a, b)

    a_grad = list()

    def grad_hook(grad):
        a_grad.append(grad)
    
    handle = a.register_hook(grad_hook)  # save a.grad
    y.backward()

    print("gradient:", w.grad, x.grad, a.grad, b.grad, y.grad)
    print("a_grad[0]: ", a_grad[0])
    handle.remove()

    w = torch.tensor([1.], requires_grad=True)
    x = torch.tensor([2.], requires_grad=True)
    a = torch.add(w, x)
    b = torch.add(w, 1)
    y = torch.mul(a, b)

    a_grad = list()

    def grad_hook_2(grad):  # what to do?
        grad *= 2
        return grad*3

    handle = w.register_hook(grad_hook_2)

    y.backward()

    print("w.grad: ", w.grad)
    handle.remove()

    # nn.Module.register_forward_hook
    # hook(module, input, output) -> None
    net = Net()
    net.conv1.weight[0].detach().fill_(1)
    net.conv1.weight[1].detach().fill_(2)
    net.conv1.bias.detach().zero_()

    fmap_block = []
    input_block = []

    def forward_hook(module, data_input, data_output):
        fmap_block.append(data_output)
        input_block.append(data_input)
    
    net.conv1.register_backward_hook(forward_hook)

    fake_img = torch.ones((1, 1, 4, 4))
    output = net(fake_img)

    print("output share:{}\noutput value:{}\n".format(output.size(), output))
    print("feature map share:{}\noutput value:{}\n".format(fmap_block[0].shape, fmap_block[0]))
    print("input share:{}\ninput value:{}\n".format(input_block[0][0].size(), input_block[0][0]))
    # please lookup nn.Module __call__()

    # nn.Module.register_forward_pre_hook
    # hook(module, input) -> None

    # nn.Module.register_backward_hook -> None
    # hook(module, grad_input, grad_output) -> Tensor or None
    def forward_pre_hook(module, data_input):
        print("forward_pre_hook input: {}".format(data_input))
    
    def backward_hook(module, grad_input, grad_output):
        print("backward hook input: {}".format(grad_input))
        print("backward hook output: {}".format(grad_output))
    
    net.conv1.register_forward_pre_hook(forward_pre_hook)
    net.conv1.register_backward_hook(backward_hook)


def hook_example():

    import torch.nn as nn
    import numpy as np
    from PIL import Image
    import torchvision.transforms as transforms
    import torchvision.utils as vutils
    from torch.utils.tensorboard import SummaryWriter
    from tools.common_tools import set_seed
    import torchvision.models as models

    set_seed(1)  # 设置随机种子

    # ----------------------------------- feature map visualization -----------------------------------
    # flag = 0
    flag = 1
    if flag:
        writer = SummaryWriter(comment='test_your_comment', filename_suffix="_test_your_filename_suffix")

        # 数据
        path_img = "./lena.png"     # your path to image
        normMean = [0.49139968, 0.48215827, 0.44653124]
        normStd = [0.24703233, 0.24348505, 0.26158768]

        norm_transform = transforms.Normalize(normMean, normStd)
        img_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            norm_transform
        ])

        img_pil = Image.open(path_img).convert('RGB')
        if img_transforms is not None:
            img_tensor = img_transforms(img_pil)
        img_tensor.unsqueeze_(0)    # chw --> bchw

        # 模型
        alexnet = models.alexnet(pretrained=True)

        # 注册hook
        fmap_dict = dict()
        for name, sub_module in alexnet.named_modules():

            if isinstance(sub_module, nn.Conv2d):
                key_name = str(sub_module.weight.shape)
                fmap_dict.setdefault(key_name, list())

                n1, n2 = name.split(".")

                def hook_func(m, i, o):
                    key_name = str(m.weight.shape)
                    fmap_dict[key_name].append(o)

                alexnet._modules[n1]._modules[n2].register_forward_hook(hook_func)

        # forward
        output = alexnet(img_tensor)

        # add image
        for layer_name, fmap_list in fmap_dict.items():
            fmap = fmap_list[0]
            fmap.transpose_(0, 1)

            nrow = int(np.sqrt(fmap.shape[0]))
            fmap_grid = vutils.make_grid(fmap, normalize=True, scale_each=True, nrow=nrow)
            writer.add_image('feature map in {}'.format(layer_name), fmap_grid, global_step=322)