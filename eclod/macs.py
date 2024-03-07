import argparse
import os
import warnings
import yaml
import sys
sys.path.append('/home/pasti/PycharmProjects/Continual_Nanodet/')

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.accelerators import find_usable_cuda_devices
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
from nanodet.data.collate import naive_collate
from nanodet.data.dataset import build_dataset
from nanodet.evaluator import build_evaluator
from nanodet.trainer.task import TrainingTask
from nanodet.trainer.latent_dist_task import LatentDistTrainingTask
from torchvision.transforms import ToTensor, ToPILImage
import thop
from nanodet.util import (
    NanoDetLightningLogger,
    cfg,
    convert_old_model,
    env_utils,
    load_config,
    load_model_weight,
    mkdir,
)
from nanodet.model.arch import build_model

load_config(cfg, 'cfg/task0.yml')
model = build_model(cfg.model)
input = torch.randn(1, 3, 320, 320)
print(model.backbone)
#layer_n_macs = model.backbone.stage4[3].parameters()
out = model.backbone.conv1(input)
input = out
out = model.backbone.maxpool(input)
input = out
out = model.backbone.stage2(input)
input = out
out = model.backbone.stage3(input)
print(out.shape)
input = torch.randn(1,232,20,20)
flops, params = thop.profile(model.backbone.stage4[0], inputs=(input, ))
out = model.backbone.stage4[0](input)
input = out
#flops, params = thop.profile(model.backbone.stage4[1], inputs=(input, ))
out = model.backbone.stage4[1](input)
input = out
#flops, params = thop.profile(model.backbone.stage4[2], inputs=(input, ))
out = model.backbone.stage4[2](input)
input = out
#flops, params = thop.profile(model.backbone.stage4[3], inputs=(input, ))
print(f'flops: {flops}, params: {params}')

#465388000
#stage4[0] = 33454400
#stage4[1] = 11252000
#stage4[2] = 11252000
#stage4[3] = 11252000 
#261637600 # BACKBONE FLOPS
#Network - backbone = 203750400
#backbone-1 = 215002400
#backbone-2 = 226254400
#backbone-3 = 237506400
#backbone-4 = 270960800
print(270960800/465388000)