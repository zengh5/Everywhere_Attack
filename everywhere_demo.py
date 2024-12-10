"""
Everywhere attack for 2025AAAI submission id: 6568
this demo shows how to integrate the proposed everywhere scheme with the CE attack
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import scipy.io as scio
import scipy.stats as st
import random
from torchvision.models import Inception_V3_Weights, ResNet50_Weights, DenseNet121_Weights, VGG16_BN_Weights
from torchvision.utils import save_image


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# simple Module to normalize an image
class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = torch.Tensor(mean)
        self.std = torch.Tensor(std)

    def forward(self, x):
        return (x - self.mean.type_as(x)[None, :, None, None]) / self.std.type_as(x)[None, :, None, None]


# define DI_keepresolution
def DI_keepresolution(X_in):
    rnd = np.random.randint(270, 299, size=1)[0]
    h_rem = 299 - rnd
    w_rem = 299 - rnd
    pad_top = np.random.randint(0, h_rem, size=1)[0]
    pad_bottom = h_rem - pad_top
    pad_left = np.random.randint(0, w_rem, size=1)[0]
    pad_right = w_rem - pad_left

    c = np.random.rand(1)
    if c <= 0.7:
        X_in_inter = F.interpolate(X_in, size=(rnd, rnd))
        X_out = F.pad(X_in_inter, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
        return X_out
    else:
        return X_in


# define TI
def gkern(kernlen=15, nsig=3):
    x = np.linspace(-nsig, nsig, kernlen)
    kern1d = st.norm.pdf(x)
    kernel_raw = np.outer(kern1d, kern1d)
    kernel = kernel_raw / kernel_raw.sum()
    return kernel


channels = 3
kernel_size = 5
kernel = gkern(kernel_size, 3).astype(np.float32)
gaussian_kernel = np.stack([kernel, kernel, kernel])
gaussian_kernel = np.expand_dims(gaussian_kernel, 1)
gaussian_kernel = torch.from_numpy(gaussian_kernel).cuda()

# 1. Model: load the pretrained models
model_1 = models.inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, transform_input=True).eval()
model_2 = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).eval()
model_3 = models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1).eval()
model_4 = models.vgg16_bn(weights=VGG16_BN_Weights.IMAGENET1K_V1).eval()

for param in model_1.parameters():
    param.requires_grad = False
for param in model_2.parameters():
    param.requires_grad = False
for param in model_3.parameters():
    param.requires_grad = False
for param in model_4.parameters():
    param.requires_grad = False

device = torch.device("cuda:0")
model_1.to(device)
model_2.to(device)
model_3.to(device)
model_4.to(device)

# 2. Data: 1000 images from the ImageNet-Compatible dataset
# values are standard normalization for ImageNet images
norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
trn = transforms.Compose([transforms.ToTensor(), ])

# 3. Parameters
batch_size = 1
max_iterations = 200
img_size = 299
lr = 2 / 255  # step size
epsilon = 16  # L_inf norm bound
##### new paras
N = 9         # the number of samples, N = 9 by default

# 4. Attacks
# CE + everywhere
setup_seed(42)
for k in range(0, 1):
    # print(k)
    X_ori = torch.zeros(batch_size, 3, img_size, img_size).to(device)
    delta = torch.zeros_like(X_ori, requires_grad=True).to(device)
    for i in range(batch_size):
        X_ori[i] = trn(Image.open('f43fbfe8a9ea876c.png'))
    labels = torch.tensor(377).unsqueeze(dim=0).to(device)   # target: 377='marmoset'
    labels_combine = labels.repeat(N+1)                      #
    #### everywhere, the number of partitions M = 4
    mask = torch.zeros(16, batch_size, 3, img_size, img_size).to(device)
    for i_mask in range(16):
        up = int(np.floor(i_mask / 4) * 75)
        down = min(up + 75, 299)
        left = int((i_mask % 4) * 75)
        right = min(left + 75, 299)

        mask[i_mask, :, :, up:down, left:right] = 1
        # plt.imshow(mask[i_mask, 0, 0].cpu(), cmap='gray')
        # plt.show()

    grad_pre = 0
    for t in range(max_iterations):
        # print(t)
        X_adv = X_ori + delta
        X_adv_DI = DI_keepresolution(X_adv)  # DI
        X_adv_norm_DI = norm(X_adv_DI)
        ## everywhere
        idx = torch.randperm(16)
        X_adv_combine = torch.zeros((N + 1) * batch_size, 3, img_size, img_size).to(device)
        X_adv_combine[:batch_size] = X_adv_norm_DI
        for i_mask in range(N):
            X_adv_combine[((i_mask + 1) * batch_size): ((i_mask + 2) * batch_size)] =\
                (mask[idx[i_mask]] * X_adv_norm_DI)

        # logits = model_1(X_adv_combine)
        # logits = model_2(X_adv_combine)
        # logits = model_3(X_adv_combine)
        logits = model_4(X_adv_combine)    # Vgg16 as the surrogate

        # CE
        loss = nn.CrossEntropyLoss(reduction='sum')(logits, labels_combine)
        # logit
        # real = logits.gather(1, labels_combine.unsqueeze(1)).squeeze(1)
        # logit_dists = (-1 * real)
        # loss = logit_dists.sum()

        loss.backward()
        grad_c = delta.grad.clone()
        grad_c = F.conv2d(grad_c, gaussian_kernel, bias=None, stride=1, padding=(2, 2), groups=3)  # TI
        # grad_a = grad_c + 1 * grad_pre  # MI for Po, logit, SupHigh
        grad_a = grad_c / torch.mean(torch.abs(grad_c), (1, 2, 3), keepdim=True) + 1 * grad_pre  # MI for CE

        grad_pre = grad_a
        delta.grad.zero_()
        delta.data = delta.data - lr * torch.sign(grad_a)
        delta.data = delta.data.clamp(-epsilon / 255, epsilon / 255)
        delta.data = ((X_ori + delta.data).clamp(0, 1)) - X_ori

    X_adv_norm = norm(X_ori + delta).detach()
    logit1 = model_1(X_adv_norm)
    logit2 = model_2(X_adv_norm)
    logit3 = model_3(X_adv_norm)

    print('Output label by Inc-v3:')
    print(torch.argmax(logit1, dim=1))
    print('Output label by Res50:')
    print(torch.argmax(logit2, dim=1))
    print('Output label by Dense121:')
    print(torch.argmax(logit3, dim=1))

torch.cuda.empty_cache()
Done = 1
