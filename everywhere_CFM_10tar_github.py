"""
CFM + everywhere in the 10-tar scenario
Note: CFM does not work for a single benign image. Hence, we need to prepare a dataset for CFM.
"""

from torchvision import datasets, models, transforms
from PIL import Image
import numpy as np
from utils_baseattack import gkern, DI, DI_keepresolution, DI_pa, Poincare_dis, Cos_dis, load_ground_truth, Cos_dis_sign
import random
import math
from torchvision.utils import save_image
from CFMgithub.attacks_every import *
import csv


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# load image metadata (Image_ID, true label, and target label)
def load_ground_truth(csv_filename):
    image_id_list = []
    label_ori_list = []
    label_tar_list = []

    with open(csv_filename) as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        for row in reader:
            image_id_list.append(row['ImageId'])
            label_ori_list.append(int(row['TrueLabel']) - 1)
            label_tar_list.append(int(row['TargetClass']) - 1)

    return image_id_list, label_ori_list, label_tar_list


# simple Module to normalize an image
class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = torch.Tensor(mean)
        self.std = torch.Tensor(std)

    def forward(self, x):
        return (x - self.mean.type_as(x)[None, :, None, None]) / self.std.type_as(x)[None, :, None, None]


## simple wrapper model to normalize an input image
class WrapperModel(nn.Module):
    def __init__(self, model, mean, std,resize=False):
        super(WrapperModel, self).__init__()
        self.mean = torch.Tensor(mean)
        self.model = model
        self.resize = resize
        self.std = torch.Tensor(std)

    def forward(self, x):
        if self.resize == True:
            x = transforms.Resize((224, 224), interpolation=InterpolationMode.NEAREST)(x)
        x_norm = (x - self.mean.type_as(x)[None, :, None, None]) / (self.std.type_as(x)[None, :, None, None])
        return self.model(x_norm)

    def normalize(self, x):
        if self.resize == True:
            x = transforms.Resize((224, 224), interpolation=InterpolationMode.NEAREST)(x)
        return (x - self.mean.type_as(x)[None,:,None,None]) / self.std.type_as(x)[None,:,None,None]


channels = 3
kernel_size = 5
kernel = gkern(kernel_size, 3).astype(np.float32)
gaussian_kernel = np.stack([kernel, kernel, kernel])
gaussian_kernel = np.expand_dims(gaussian_kernel, 1)
gaussian_kernel = torch.from_numpy(gaussian_kernel).cuda()

# 1. Model: load the pretrained models
# source model: model_2; target models: model_1, 3, 4
model_2 = models.resnet50(pretrained=True).eval()

for param in model_2.parameters():
    param.requires_grad = False

device = torch.device("cuda:0")
model_2.to(device)

torch.backends.cudnn.deterministic = True

# 2. Data: 1000 images from the ImageNet-Compatible dataset
# values are standard normalization for ImageNet images,
# from https://github.com/pytorch/examples/blob/master/imagenet/main.py
norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
trn = transforms.Compose([transforms.ToTensor(), ])

# 3. Parameters
batch_size = 5
max_iterations = 200
# input_path = '../Target/dataset/images/'
input_path = './dataset/images/'
img_size = 299
epsilon = 16  # L_inf norm bound
lr = 2 / 255  # step size

targetlabel_list = [24, 99, 245, 344, 471, 555, 661, 701, 802, 919]

# pre-process input image
mean, stddev = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
source_model = WrapperModel(model_2, mean, stddev).to(device)
source_model = source_model.eval()

for i_tar in range(0, 10):
    targetlabel = targetlabel_list[i_tar]
    output_path = 'adv_imgs/' + str(targetlabel) + '/'
    image_id_list, label_ori_list, label_tar_list = load_ground_truth('./dataset/images.csv')
    # Remove samples that belong to the target attack label.
    label_ori_list_t = []
    image_id_list_t = []
    for ii in range(0, 100):
        if label_ori_list[ii] != targetlabel:
            label_ori_list_t.append(label_ori_list[ii])
            image_id_list_t.append(image_id_list[ii])
    image_id_list = image_id_list_t
    label_ori_list = label_ori_list_t
    test_size = len(label_ori_list)
    num_batches = int(math.ceil(test_size / batch_size))

    # 4. Attack
    setup_seed(42)
    for k in range(0, num_batches):
        ###
        source_model.eval()
        batch_size_cur = min(batch_size, len(image_id_list) - k * batch_size)
        img = torch.zeros(batch_size_cur, 3, img_size, img_size).to(device)
        id_cur = k * batch_size + batch_size_cur
        target_labels = torch.tensor(label_tar_list[k * batch_size: id_cur]).to(device)
        image_id = []
        for i in range(batch_size_cur):
            img[i] = trn(Image.open(input_path + image_id_list[k * batch_size + i] + '.png'))
            image_id.append(image_id_list[k * batch_size + i])
            target_labels[i] = targetlabel
        labels = torch.tensor(label_ori_list[k * batch_size: id_cur]).to(device)
        # Generate adversarial examples
        # everywhere attack
        x_adv = advanced_fgsm_every_memory(attack_type='CDTM', source_model=source_model, x=img, y=labels,
                                           target_label=target_labels, num_iter=200, max_epsilon=16, count=k, config_idx=2)

        # for j in range(batch_size_cur):
        #     save_image(X_adv[j].data, output_path + image_id[j] + ".png")
        #############
Done = 1



