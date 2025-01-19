
import torch
import os
import numpy as np
import scipy.stats as st
import torch.nn.functional as F
from torchvision.transforms import InterpolationMode
from typing import Callable
from CFMgithub.config import *
from torch import nn, Tensor

##########################################
device = torch.device("cuda")
dir_path = os.path.dirname(os.path.realpath(__file__))


# define DI, keep resolution
def DI_keepresolution(X_in):
    img_size = X_in.shape[3]
    rnd = np.random.randint(img_size-29, img_size, size=1)[0]
    h_rem = img_size - rnd
    w_rem = img_size - rnd
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


def gkern(kernlen=15, nsig=3):
    x = np.linspace(-nsig, nsig, kernlen)
    kern1d = st.norm.pdf(x)
    kernel_raw = np.outer(kern1d, kern1d)
    kernel = kernel_raw / kernel_raw.sum()
    return kernel


class LogitLoss(nn.Module):
    def __init__(self, labels, targeted=True):
        super(LogitLoss, self).__init__()
        self.labels = labels
        self.targeted = targeted
        self.labels.requires_grad = False

    def forward(self, logits):
        real = logits.gather(1, self.labels.unsqueeze(1)).squeeze(1)
        logit_dists = (1 * real)
        loss = logit_dists.sum()
        if self.targeted == False:
            loss = -loss
        return loss


# everywhere Feature Mixup
class FeatureMixupEverywhere(nn.Module):
    def __init__(self, model: nn.Module, config_idx, input_size):
        super().__init__()
        exp_settings = exp_configuration[config_idx]
        self.mixup_layer = exp_settings['mixup_layer']
        self.prob = exp_settings['mix_prob']
        self.channelwise = exp_settings['channelwise']

        self.model = model
        self.input_size = input_size
        self.record = False
        self.outputs = {}
        self.forward_hooks = []
        self.batchsize = 1
        self.masknum = 0
        self.selected_region = []

        def get_children(model: torch.nn.Module):
            children = list(model.children())
            flattened_children = []
            if children == []:
                # if model is the last child
                if self.mixup_layer == 'conv_linear_no_last' or self.mixup_layer == 'conv_linear_include_last':
                    if type(model) == torch.nn.Conv2d or type(model) == torch.nn.Linear:
                        return model
                    else:
                        return []
                elif self.mixup_layer == 'bn' or self.mixup_layer == 'relu':
                    if type(model) == torch.nn.BatchNorm2d:
                        return model
                    else:
                        return []
                else:
                    if type(model) == torch.nn.Conv2d:
                        return model
                    else:
                        return []
            else:
                # look for children
                for child in children:
                    try:
                        flattened_children.extend(get_children(child))
                    except TypeError:
                        flattened_children.append(get_children(child))
            return flattened_children

        mod_list = get_children(model)
        self.layer_num = len(mod_list)
        # print(mod_list)

        for i, m in enumerate(mod_list):
            self.forward_hooks.append(m.register_forward_hook(self.save_outputs_hook(i, config_idx)))

    def save_outputs_hook(self, layer_idx, config_idx) -> Callable:
        # Load experiment configurations
        exp_settings = exp_configuration[config_idx]
        mix_upper_bound_feature = exp_settings['mix_upper_bound_feature']
        mix_lower_bound_feature = exp_settings['mix_lower_bound_feature']
        shuffle_image_feature = exp_settings['shuffle_image_feature']
        blending_mode_feature = exp_settings['blending_mode_feature']
        mixed_image_type_feature = exp_settings['mixed_image_type_feature']
        divisor = exp_settings['divisor']

        def hook_fn(module, input, output):
            if type(module) == torch.nn.Linear or output.size()[-1] <= self.input_size // divisor:

                if self.mixup_layer == 'conv_linear_no_last' and (layer_idx + 1) == self.layer_num and type(
                        module) == torch.nn.Linear:
                    pass  # exclude the last fc layer
                else:
                    if layer_idx in self.outputs and self.record == False:  # Feature mixup inference mode
                        c = torch.rand(1).item()
                        if c <= self.prob:  # With probability p
                            if mixed_image_type_feature == 'A':  # Mix features of other images
                                prev_feature = output.clone().detach()
                            else:  # Mix clean features
                                prev_feature = self.outputs[layer_idx].clone().detach()  # Get stored clean features

                            if shuffle_image_feature == 'SelfShuffle':  # Image-wise feature shuffling
                                # CFM is the same across different regions
                                idx0 = torch.randperm(self.batchsize)
                                idx = idx0.clone()
                                for i_region in range(len(self.selected_region)):
                                    idx = torch.cat([idx, idx0 + (self.selected_region[i_region] + 1) * self.batchsize], dim=0)

                                total_sample = (len(self.selected_region) + 1) * self.batchsize
                                prev_feature_shuffle = prev_feature[idx].view(prev_feature[:total_sample].size())
                                del idx
                            elif shuffle_image_feature == 'None':
                                prev_feature_shuffle = prev_feature

                            # Random mixing ratio
                            mix_ratio = mix_upper_bound_feature - mix_lower_bound_feature
                            num_samples = prev_feature_shuffle.shape[0]
                            num_channels = prev_feature_shuffle.shape[1]
                            if self.channelwise == True:
                                if output.dim() == 4:
                                    a = ((torch.rand(num_samples, num_channels) * mix_ratio + mix_lower_bound_feature)
                                         .view(num_samples, num_channels, 1, 1).cuda())
                                elif output.dim() == 3:
                                    a = ((torch.rand(num_samples, num_channels) * mix_ratio + mix_lower_bound_feature)
                                         .view(num_samples, num_channels, 1).cuda())
                                else:
                                    a = ((torch.rand(num_samples, num_channels) * mix_ratio + mix_lower_bound_feature)
                                         .view(num_samples, num_channels).cuda())
                            else:  # image by image, easy to understand
                                if output.dim() == 4:
                                    a = (torch.rand(num_samples) * mix_ratio + mix_lower_bound_feature).view(
                                        num_samples, 1, 1, 1).cuda()
                                elif output.dim() == 3:
                                    a = (torch.rand(num_samples) * mix_ratio + mix_lower_bound_feature).view(
                                        num_samples, 1, 1).cuda()
                                else:
                                    a = (torch.rand(num_samples) * mix_ratio + mix_lower_bound_feature).view(
                                        num_samples, 1).cuda()
                            # Blending
                            if self.mixup_layer == 'relu':
                                output = F.relu(output, inplace=True)
                            # Core code
                            if blending_mode_feature == 'M':  # Linear interpolation
                                output2 = (1 - a) * output + a * prev_feature_shuffle
                            elif blending_mode_feature == 'A':  # Addition
                                output2 = output + a * prev_feature_shuffle

                            return output2
                        else:
                            return output

                    elif self.record == True:  # Feature recording mode
                        self.outputs[layer_idx] = output.clone().detach()
                        return

        return hook_fn

    def start_feature_record(self):
        self.record = True

    def end_feature_record(self):
        self.record = False

    def set_paras(self, batchsize, masknum, selected_region):
        self.batchsize = batchsize
        self.masknum = masknum
        self.selected_region = selected_region

    def remove_hooks(self):
        for fh in self.forward_hooks:
            fh.remove()
        del self.outputs

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)     # jump to forward() in utils.py


class CELoss(nn.Module):
    def __init__(self, labels):
        super(CELoss, self).__init__()
        self.labels = labels
        self.ce = nn.CrossEntropyLoss(reduction='mean')
        self.labels.requires_grad = False

    def forward(self, logits):
        return self.ce(logits, self.labels)


# memory-efficient version of CFM + everywhere, sample 4 from 9 blocks
# In the paper, we sample 9 from 16 blocks
def advanced_fgsm_every_memory(attack_type, source_model, x, y, target_label=None, num_iter=10, max_epsilon=16, mu=1.0,
                             count=0, config_idx=1):
    """CFM+everywhere
    """
    #########
    sample_num = 4
    mask = torch.zeros(9, x.shape[0], 3, 299, 299).to(device)
    for i_mask in range(9):
        up = int(np.floor(i_mask / 3) * 100)
        down = min(up + 100, 299)
        left = int((i_mask % 3) * 100)
        right = min(left + 100, 299)

        mask[i_mask, :, :, up:down, left:right] = 1

    labels_combine = torch.cat([target_label, target_label, target_label, target_label, target_label], dim=0)
    ######
    # Load experiment configurations
    delta = torch.zeros_like(x, requires_grad=True).to(device)

    exp_settings = exp_configuration[config_idx]
    prob = exp_settings['p']
    lr = exp_settings['alpha']  # Step size eta
    if 'targeted' not in exp_settings:
        exp_settings['targeted'] = True
    if "M" not in attack_type and "N" not in attack_type:
        mu = 0

    ti_kernel_size = 5
    if 'T' in attack_type:  # Tranlation-invariance
        kernel = gkern(ti_kernel_size, 3).astype(np.float32)
        gaussian_kernel = np.stack([kernel, kernel, kernel])
        gaussian_kernel = np.expand_dims(gaussian_kernel, 1)
        gaussian_kernel = torch.from_numpy(gaussian_kernel).cuda()
    source_model.eval()

    eps = max_epsilon / 255.0  # epsilon in scale [0, 1]
    alpha = lr / 255.0

    g = 0    # g_previous
    # Set loss function
    if exp_settings['targeted']:
        loss_fn = LogitLoss(labels_combine, exp_settings['targeted'])

    B, C, H, W = x.size()
    # Memory for generated adversarial examples
    # x_advs = torch.zeros((num_iter // save_interval, B, C, H, W)).to(device)
    ##########  mean_tensor is the normalized tensor
    mean = [0.485, 0.456, 0.406]
    mean_tensor = torch.Tensor(mean).type_as(x)[None, :, None, None] * torch.ones_like(x)

    consumed_iteration = 0
    if 'C' in attack_type:  # Storing clean features at the first iteration
        with torch.no_grad():
            img_width = x.size()[-1]  # B X C X H X W
            ###
            # 4 by 4
            X_combine = torch.zeros((9 + 1) * x.shape[0], 3, 299, 299).to(device)
            X_combine[:x.shape[0]] = x  # the whole image

            for i_mask in range(9):
                X_combine[((i_mask + 1) * x.shape[0]): ((i_mask + 2) * x.shape[0])] = \
                    (mask[i_mask] * x) + ((1-mask[i_mask]) * mean_tensor)
            #######
            # hooks are used to mixup features
            model = FeatureMixupEverywhere(source_model, config_idx, img_width)  # Attach CFM modules to conv and fc layers

            model.start_feature_record()  # Set feature recoding mode
            # model(x_f)                  # Feature recording
            model(X_combine)              # Feature recording
            model.end_feature_record()  # Set feature mixup inference mode

            consumed_iteration = 1  # Deduct 1 iteration in total iterations for strictly fair comparisons
    else:
        model = source_model

    for t in range(num_iter):
        x_adv = x + delta
        # 3 by 3
        X_adv_combine = torch.zeros((sample_num + 1) * x.shape[0], 3, 299, 299).to(device)
        X_adv_combine[:x.shape[0]] = x_adv                   # the whole image
        if t >= consumed_iteration:
            idx = torch.randperm(9)
            model.set_paras(batchsize=x.shape[0], masknum=9, selected_region=idx[:4])
            ###
            for i_mask in range(sample_num):
                X_adv_combine[((i_mask + 1) * x.shape[0]): ((i_mask + 2) * x.shape[0])] = \
                    (mask[idx[i_mask]] * x_adv) + ((1-mask[idx[i_mask]]) * mean_tensor)
            #######
            # usual momentum
            x_nes = X_adv_combine

            if 'D' in attack_type:
                x_adv_or_nes = DI_keepresolution(x_nes)
            else:
                x_adv_or_nes = x_nes
            # ！！ forward() has been modified by hook_fn（）
            output2 = model(x_adv_or_nes)
            loss = loss_fn(output2)
            # print(loss.data)
            ghat = torch.autograd.grad(loss, delta, retain_graph=False, create_graph=False)[0]
            # Update g
            grad_plus_v = ghat
            if 'T' in attack_type:  # Translation-invariance
                grad_plus_v = F.conv2d(grad_plus_v, gaussian_kernel, bias=None, stride=1,
                                       padding=((ti_kernel_size - 1) // 2, (ti_kernel_size - 1) // 2), groups=3)  # TI
            if 'M' in attack_type or 'N' in attack_type:
                g = mu * g + grad_plus_v / torch.sum(torch.abs(grad_plus_v), dim=[1, 2, 3], keepdim=True)
            else:
                g = grad_plus_v
            # Update x_adv
            pert = alpha * g.sign()
            delta.data = delta.data + pert
            delta.data = delta.data.clamp(-eps, eps)
            delta.data = ((x + delta.data).clamp(0, 1)) - x

        x_adv = (x + delta).detach()  # zh
        # if (t + 1) % save_interval == 0:
        #     x_advs[(t + 1) // save_interval - 1] = x_adv.clone().detach()

    if 'C' in attack_type:
        model.remove_hooks()
    torch.cuda.empty_cache()
    return x_adv
