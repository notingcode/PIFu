import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import time
import json
import numpy as np
import torch
from torch.utils.data import DataLoader

from lib.options import BaseOptions
from lib.mesh_util import *
from lib.sample_util import *
from lib.train_util import *
from lib.model import *

from PIL import Image, ImageOps
import torchvision.transforms as transforms
import glob
import tqdm

# get options
opt = BaseOptions().parse()

class Evaluator:
    def __init__(self, opt):
        self.opt = opt
        self.load_size = self.opt.loadSize
        self.to_tensor = transforms.Compose([
            transforms.Resize(self.load_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        # set cuda
        if torch.cuda.is_available():
            device_name = f'cuda:{opt.gpu_id}'
        elif torch.backends.mps.is_available():
            device_name = 'mps'
        else:
            device_name = 'cpu'
        
        torch_device = torch.device(device_name)
        # set projection mode
        projection_mode = opt.projection_mode

        # create net
        netG = HGPIFuNet(opt, projection_mode).to(device=torch_device)
        print('Using Network: ', netG.name)

        if opt.load_netG_checkpoint_path:
            netG.load_state_dict(torch.load(opt.load_netG_checkpoint_path, map_location=torch_device))

        if opt.load_netC_checkpoint_path is not None:
            print('loading for net C ...', opt.load_netC_checkpoint_path)
            netC = ResBlkPIFuNet(opt).to(device=torch_device)
            netC.load_state_dict(torch.load(opt.load_netC_checkpoint_path, map_location=torch_device))
        else:
            netC = None

        os.makedirs(opt.results_path, exist_ok=True)
        os.makedirs('%s/%s' % (opt.results_path, opt.name), exist_ok=True)

        opt_log = os.path.join(opt.results_path, opt.name, 'opt.txt')
        with open(opt_log, 'w') as outfile:
            outfile.write(json.dumps(vars(opt), indent=2))

        self.device = torch_device
        self.netG = netG
        self.netC = netC

    def load_image(self, image_path, mask_path):
        # Name
        img_name = os.path.splitext(os.path.basename(image_path))[0]
        # Calib
        B_MIN = np.array(self.opt.b_min)
        B_MAX = np.array(self.opt.b_max)
        projection_matrix = np.identity(4)
        projection_matrix[1, 1] = -1
        calib = torch.Tensor(projection_matrix).float()
        transforms_matrix = torch.Tensor([[1/512,0,0],
                                          [0,1/512,0]]).float()

        # image and mask
        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        original_img_width, original_img_height = image.size

        max_dim = max(original_img_width, original_img_height)

        mask = ImageOps.pad(mask, (max_dim, max_dim))
        image = ImageOps.pad(image, (max_dim, max_dim))

        mask = ImageOps.fit(
            mask, (self.load_size, self.load_size), method=Image.NEAREST)
        image = ImageOps.fit(
            image, (self.load_size, self.load_size), method=Image.BILINEAR)

        mask.save("mask.png")
        image.save("image.png")

        mask = transforms.Resize(self.load_size)(mask)
        mask = transforms.ToTensor()(mask).float()

        image = self.to_tensor(image)
        image = mask.expand_as(image) * image

        return {
            'name': img_name,
            'img': image.unsqueeze(0),
            'calib': calib.unsqueeze(0),
            'transforms': transforms_matrix.unsqueeze(0),
            'mask': mask.unsqueeze(0),
            'b_min': B_MIN,
            'b_max': B_MAX,
        }

    def eval(self, data, use_octree=False):
        '''
        Evaluate a data point
        :param data: a dict containing at least ['name'], ['image'], ['calib'], ['b_min'] and ['b_max'] tensors.
        :return:
        '''
        opt = self.opt
        with torch.no_grad():
            self.netG.eval()
            if self.netC:
                self.netC.eval()
            save_path = '%s/%s/result_%s.obj' % (opt.results_path, opt.name, data['name'])
            if self.netC:
                gen_mesh_color(opt, self.netG, self.netC, self.device, data, save_path, use_octree=use_octree)
            else:
                gen_mesh(opt, self.netG, self.device, data, save_path, use_octree=use_octree)


if __name__ == '__main__':
    evaluator = Evaluator(opt)

    test_images = glob.glob(os.path.join(opt.test_folder_path, '*'))
    test_images = [f for f in test_images if ('png' in f or 'jpg' in f) and (not 'mask' in f)]
    test_masks = [f[:-4]+'_mask.png' for f in test_images]

    print("num; ", len(test_masks))

    for image_path, mask_path in tqdm.tqdm(zip(test_images, test_masks)):
        try:
            print(image_path, mask_path)
            data = evaluator.load_image(image_path, mask_path)
            evaluator.eval(data, True)
        except Exception as e:
           print("error:", e.args)
