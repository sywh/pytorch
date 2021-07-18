import numpy as np
import torch
import torch.nn as nn
import ipdb


# Create
def create():
    # torch.tensor(data, dtype=None, device=None, requires_grad=False, pin_memory=False)
    x1 = torch.tensor([1.0, 2.0])  # dtype=torch.float32

    # torch.from_numpy(ndarray)
    x2_np = np.array([1.0, 2.0])  
    x2 = torch.from_numpy(x2_np)  # dtype=torch.float64

    # torch.zeros(*size, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)
    x3 = torch.zeros((3, 224, 224))

    # torch.zeros_like(input, dtype=None, layout=None, device=None, requires_grad=False)
    x4 = torch.zeros_like(x3)

    # torch.ones(*size, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)
    # torch.ones_like(input, dtype=None, layout=None, device=None, requires_grad=False)

    # torch.arange(start=0, end, step=1, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)
    x5 = torch.arange(0, 10)  # 1-D tensor

    # torch.linspace(start, end, steps=100, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)
    x6 = torch.linspace(0, 1)

    # torch.logspace(start, end, steps=100, base=10.0, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)
    x7 = torch.logspace(1, 10)

    # torch.normal(mean, std, out=None)
    x8 = torch.normal(torch.tensor([0.0]), torch.tensor([1.0]))

    # torch.randn(*size, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)
    x9 = torch.randn((3, 224, 224))

    # torch.randn_like(input)
    x10 = torch.randn_like(x9)

    # torch.rand(*size, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)
    x11 = torch.rand((3, 224, 224))  # uniform in [0, 1)
    # torch.rand_like(input)

    # torch.randint(low=0, high, size, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)
    x12 = torch.randint(0, 10, (3, 224, 224))  # uniform in [low, high)
    # torch.randint_like(input)

# Operate
def operate():
    # torch.cat(tensors, dim=0, out=None)
    t = torch.ones((2, 3))
    t_cat_0 = torch.cat([t, t], dim=0)
    t_cat_1 = torch.cat([t, t], dim=1)

    # torch.stack(tensors, dim=0, out=None)  # create new dim
    t_stack_0 = torch.stack([t, t], dim=0)
    t_stack_2 = torch.stack([t, t], dim=2)

    # torch.chunk(input, chunks, dim=0)  # uniform split
    t = torch.ones((2, 5))
    t_chunk = torch.chunk(t, 2, 1)

    # torch.split(tensor, split_size_or_sections, dim=0)
    t = torch.ones((2, 5))
    t_split_1 = torch.split(t, 2, 1)  # split to 2, 2, 1
    t_split_2 = torch.split(t, [2, 1, 2], 1)  # split to 2, 1, 2

    # torch.index_select(input, dim, index, out=None)
    t = torch.randint(0, 9, size=(3, 3))
    idx = torch.tensor([0, 2], dtype=torch.long)  # must be torch.long
    t_select = torch.index_select(t, dim=0, index=idx)  # does not share memory with t

    # torch.masked_select(input, mask, out=None)
    mask = t.ge(5)
    t_select = torch.masked_select(t, mask)  # 1-dim

    # torch.transpose(input, dim0, dim1)
    t_transpose = torch.transpose(t, 0, 1)
    # torch.t() is the same as torch.transpose(input, 0, 1)

    # torch.squeeze(input, dim=None, out=None)
    t_unsqueeze = torch.unsqueeze(t, 0)
    t_squeeze = torch.squeeze(t_unsqueeze)
    # torch.unsqueeze(input, dim, out=None)

    # mathmatical calculation
    """
    torch.add()
    torch.addcdiv()
    torch.addcmul()
    torch.sub()
    torch.div()
    torch.mul()

    torch.log(input, out=None)
    torch.log10(input, out=None)
    torch.log2(input, out=None)
    torch.exp(input, out=None)
    torch.pow()

    torch.abs(input, out=None)
    torch.acos(input, out=None)
    torch.cosh(input, out=None)
    torch.cos(input, out=None)
    torch.asin(input, out=None)
    torch.atan(input, out=None)
    torch.atan2(input, ohter, out=None)
    """

    # torch.add(input, alpha=1, other, out=None)  # input + alpha * other
    x = torch.tensor([1.0, 2.0])
    y = torch.tensor([3.0, 4.0])
    z = torch.add(x, y)
    # torch.addcmul(input, value=1, tensor1, tensor2, out=None)  # input + value * tensor1 * tensor2


# Linear Regression
def linear_regression():
    x = torch.rand(20, 1) * 20
    y = 2*x + (5 + torch.randn(20, 1))
    w = torch.randn((1), requires_grad=True)
    b = torch.zeros((1), requires_grad=True)
    for iteration in range(100):
        wx = torch.mul(w, x)
        y_pred = torch.add(wx, b)
        loss = (0.5*(y-y_pred)**2).mean()
        loss.backward()

        b.data.sub_(lr * b.grad)
        w.data.sub_(lr * w.grad)

        if iteration % 20 == 0:
            plt.scatter(x.data.numpy(), y.data.numpy())
            plt.plot(x.data.numpy(), y_pred.data.numpy(), 'r-', lw=5)
            plt.text(2, 20, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size':20, 'color':'red'})
            plt.xlim(1.5, 10)
            plt.ylim(8, 28)
            plt.title("Iteration:{}\nw:{} b:{}".format(iteration, w.data.numpy(), b.data.numpy()))

            if loss.data.numpy() < 1:
                break


# Autograd
def autograd():
    # torch.autograd.grad(outputs, inputs, grad_outputs=None, retrain_graph=None, create_graph=False)
    x = torch.tensor([3.], requires_grad=True)
    y = torch.pow(x, 2)
    grad_1 = torch.autograd.grad(y, x, create_graph=True)
    grad_2 = torch.autograd.grad(grad_1[0], x)


# Logistic Regression
class LR(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Linear(2, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.features(x)
        x = self.sigmoid(x)
        return x

def logistic_regression():
    sample_nums=100
    mean_value=1.7
    bias=1
    n_data = torch.ones(sample_nums, 2)
    x0 = torch.normal(mean_value * n_data, 1) + bias
    y0 = torch.zeros(sample_nums)
    x1 = torch.normal(-mean_value * n_dta, 1) + bias
    y1 = torch.ones(sample_nums)
    train_x = torch.cat((x0, x1), 0)
    train_y = torch.cat((y0, y1), 0)

    lr_net = LR()
    loss_fn = nn.BCELoss()
    
    lr = 0.01
    optimizer = torch.optim.SGD(lr_net.parameters(), lr=lr, momentum=0.5)

    for iteration in range(1000):
        y_pred = lr_net(train_x)
        loss = loss_fn(y_pred.squeeze(), tran_y)
        loss.backward()
        optimizer.step()

        if iteration % 20 == 0:
            mask = y_pred.ge(0.5).float.squeeze()
            correct = (mask == train_y).sum()
            acc = correct.item() / train_y.size(0)

            plt.scatter(x0.data.numpy()[:, 0], x0.data.numpy()[:, 1], c='r', label='class 0')
            plt.scatter(x1.data.numpy()[:, 0], x1.data.numpy()[:, 1], c='b', label='class 1')

            w0, w1 = lr_net.features.weight[0]
            w0, w1 = float(w0.item()), float(w1.item())
            plot_b = float(lr_net.features.bias[0].item())
            polt_x = np.arange(-6, 6, 0.1)
            plot_y = (-plot_b - w0 * plot_x) / w1
            plt.xlim(-5, 7)
            plt.ylim(-7, 7)
            plt.plot(plot_x, plot_y)

            plt.text(-5, 5, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size':20, 'color':'red'})
            plt.title("Iteration:{}\nw0:{:.2f} w1:{:.2f} b:{:.2f} accuracy:{:.2%}".format(iteration, w0, w1, plot_b, acc))
            plt.legend()
            
            plt.show()
            plt.pause(0.5)

            if acc > 0.99:
                break



if __name__ == "__main__":
    operate()