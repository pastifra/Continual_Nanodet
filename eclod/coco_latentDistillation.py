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
from nanodet.util import (
    NanoDetLightningLogger,
    cfg,
    convert_old_model,
    env_utils,
    load_config,
    load_model_weight,
    mkdir,
)

#Set logger and seed
logger = NanoDetLightningLogger('test')
pl.seed_everything(9)

#Function to create the task configuration file required for training
def create_exp_cfg(yml_path, task):
    all_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
                 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 
                 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 
                 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 
                 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 
                 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 
                 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 
                 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 
                 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
    #Load the YAML file
    with open(yml_path, 'r') as file:
        temp_cfg = yaml.safe_load(file)
    #Save dir of the model
    temp_cfg['save_dir'] = 'models/COCOtask' + str(task)
    #If base task, training and testing classes are the same
    if task == 0:
        temp_cfg['data']['train']['exp_names'] = all_names[:40]
        temp_cfg['data']['val']['exp_names'] = all_names[:40]
        temp_cfg['model']['arch']['head']['num_classes'] = 80
    #Else, training only on task specific class, and testing on all classes
    else:
        start = 40 + (task - 1) * 10 + 1 #Going 10 classes at a time
        end = start + 10
        temp_cfg['data']['train']['exp_names'] = all_names[start:end]
        temp_cfg['data']['val']['exp_names'] = all_names[:end]
        temp_cfg['model']['arch']['head']['num_classes'] = 80
        temp_cfg['schedule']['load_model'] = 'models/COCOtask' + str(task-1) + '/model_last.ckpt'
    #Set the learning rate depdending on the task dataset
    temp_cfg_name = 'cfg/COCOtask' + str(task) + '.yml'
    #Save the new configuration file
    with open(temp_cfg_name, 'w') as file:
        yaml.safe_dump(temp_cfg, file)

for task in range (0, 6):
    logger = NanoDetLightningLogger('run_logs/task'+str(task))
    logger.info("Starting task" + str(task))
    logger.info("Setting up data...")
    #Create the task configuration file based on the task number and load the configuration
    create_exp_cfg('cfg/COCO_dist.yml', task)
    load_config(cfg, 'cfg/COCOtask' + str(task) + '.yml')
    #Build datasets and dataloaders based on the task configuration file
    train_dataset = build_dataset(cfg.data.train, "train")
    #If task is not 0, create the replay dataset using the buffer
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.device.batchsize_per_gpu,
        shuffle=True,
        num_workers=cfg.device.workers_per_gpu,
        pin_memory=True,
        collate_fn=naive_collate,
        drop_last=True,
    )
    val_dataset = build_dataset(cfg.data.val, "test")
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.device.batchsize_per_gpu,
        shuffle=False,
        num_workers=cfg.device.workers_per_gpu,
        pin_memory=True,
        collate_fn=naive_collate,
        drop_last=False,
    )
    evaluator = build_evaluator(cfg.evaluator, val_dataset)
    
    #Create the model based on the task configuration file
    logger.info("Creating models")
    if task == 0:
        TrainTask = TrainingTask(cfg, evaluator)
        for param in TrainTask.model.backbone.parameters():
            param.requires_grad = False
    else:
        TrainTask = LatentDistTrainingTask(cfg, evaluator)
        #Load the model weights if task is not 0
        if "load_model" in cfg.schedule:
            ckpt = torch.load(cfg.schedule.load_model)
            load_model_weight(TrainTask.model, ckpt, logger)
            load_model_weight(TrainTask.teacher, ckpt, logger)
            logger.info("Loaded model weight from {}".format(cfg.schedule.load_model))
    
    model_resume_path = (
        os.path.join(cfg.save_dir, "model_last.ckpt")
        if "resume" in cfg.schedule
        else None
    )
    #Set the device to GPU if available
    if cfg.device.gpu_ids == -1:
        logger.info("Using CPU training")
        accelerator, devices, strategy, precision = (
            "cpu",
            None,
            None,
            cfg.device.precision,
        )
    else:
        accelerator, devices, strategy, precision = (
            "gpu",
            cfg.device.gpu_ids,
            None,
            cfg.device.precision,
        )

    if devices and len(devices) > 1:
        strategy = "ddp"
        env_utils.set_multi_processing(distributed=True)

    trainer = pl.Trainer(
        default_root_dir=cfg.save_dir,
        max_epochs=cfg.schedule.total_epochs,
        check_val_every_n_epoch=cfg.schedule.val_intervals,
        accelerator=accelerator,
        devices=[1],
        log_every_n_steps=cfg.log.interval,
        num_sanity_val_steps=0,
        callbacks=[TQDMProgressBar(refresh_rate=0)],
        logger=logger,
        benchmark=cfg.get("cudnn_benchmark", True),
        gradient_clip_val=cfg.get("grad_clip", 0.0),
        strategy=strategy,
        precision=precision,
    )
    trainer.fit(TrainTask, train_dataloader, val_dataloader, ckpt_path=model_resume_path)
    state_dict = TrainTask.model.state_dict()
    new_state_dict = {k: v for k, v in state_dict.items() if "teacher" not in k}
    
    torch.save({'state_dict': new_state_dict}, 'models/COCOtask' + str(task) + '/model_last.ckpt')