import torch.nn as nn
from torch import autograd
import numpy as np
from PIL import Image
import os

def gradient_penalty(real_pred, real_img):
    (grad_real,) = autograd.grad(
        outputs=real_pred.sum(), inputs=real_img, create_graph=True
    )
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()
    return grad_penalty

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()

    def forward(self, x):
        # Do your print / debug stuff here
        print(x.size())
        return x

class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


def get_tensor_info(tensor):
    info = []
    for name in ['requires_grad', 'is_leaf', 'retains_grad', 'grad_fn', 'grad']:
        info.append(f'{name}({getattr(tensor, name, None)})')
    info.append(f'tensor({str(tensor)})')
    return ' '.join(info)


def tensor2im(image_tensor, imtype=np.uint8):
    # print(image_tensor.size())
    image_numpy = image_tensor.cpu().float().numpy()
    # print(image_numpy.shape)
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    image_numpy = np.clip(image_numpy, 0, 255)
    return image_numpy.astype(imtype)


def img_save(image_tensor, img_type, iteration, img_path):
    image_tensor = image_tensor.squeeze(0).detach()
    img = Image.fromarray(tensor2im(image_tensor))
    img.save(os.path.join(img_path, f'{img_type}_{str(iteration)}.png'))

def timer(start,end):
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    time_elapsed = '{:0>2}:{:0>2}:{:05.2f}'.format(int(hours),int(minutes),seconds)
    return time_elapsed