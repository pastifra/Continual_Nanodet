import argparse
import os
import warnings
import yaml

import sys
sys.path.append('/home/pasti/PycharmProjects/Continual_Nanodet/') #Change this to the path of the cloned repository

import pytorch_lightning as pl
import torch
import torch.nn.init as init
import torch.nn as nn
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.accelerators import find_usable_cuda_devices
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
from nanodet.data.collate import naive_collate
from nanodet.data.dataset import build_dataset
from nanodet.evaluator import build_evaluator
from nanodet.trainer.task import TrainingTask
from nanodet.trainer.LwF_task import DistTrainingTask
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


#Function to create the task configuration file required for training
def create_exp_cfg(yml_path, task, CLtask):
    all_names = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
    #Load the YAML file
    with open(yml_path, 'r') as file:
        temp_cfg = yaml.safe_load(file)
    #Save dir of the model
    temp_cfg['save_dir'] = 'models/' + CLtask + 'task' + str(task)
    
    #Define parameters based on the CL task
    if CLtask == '10p10':
        x = 10
        train_names = all_names[10:]
        val_names = all_names
    elif CLtask == '19p1' :
        x = 19
        train_names = [all_names[19]]
        val_names = all_names
    elif CLtask == '15p5' :
        x = 15
        train_names = all_names[15:]
        val_names = all_names
    elif CLtask == '15p1' :
        x = 15
        train_names = [all_names[14+task]]
        val_names = all_names[:15+task]
    #If base task, training and testing classes are the same
    if task == 0:
        temp_cfg['data']['train']['class_names'] = all_names[:x]
        temp_cfg['data']['val']['class_names'] = all_names[:x]
    #Else, training only on task specific class, and testing on all classes
    else:
        temp_cfg['data']['train']['class_names'] = train_names
        temp_cfg['data']['val']['class_names'] = val_names
        temp_cfg['schedule']['load_model'] = 'models/' + CLtask + 'task' + str(task-1) + '/model_last.ckpt'
        
    temp_cfg_name = 'cfg/task' + str(task) + '.yml'
    print(temp_cfg_name)
    #Save the new configuration file
    with open(temp_cfg_name, 'w') as file:
        yaml.safe_dump(temp_cfg, file)

#Set logger and seed
logger = NanoDetLightningLogger('test')
pl.seed_everything(1234)

parser = argparse.ArgumentParser(description="Parser for training task")
parser.add_argument('--task', type=str, help='Type of CL task to train (15p5, 10p10, 19p1, 15p1)', required=True)
parser.add_argument('--cfg', type=str, help='Path to the configuration file', required=True)
parser.add_argument('--alpha', type=float, help='Weight for distillation loss', default=1.0)

args = parser.parse_args()

if args.task == '15p1':
    total_tasks = 6
else:
    total_tasks = 2

for task in range (0, total_tasks):
    logger = NanoDetLightningLogger('run_logs/task'+str(task))
    logger.info("Starting task" + str(task))
    logger.info("Setting up data...")
    #Create the task configuration file based on the task number and load the configuration
    create_exp_cfg(args.cfg, task, args.task)
    load_config(cfg, 'cfg/task' + str(task) + '.yml')
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
    else:
        TrainTask = DistTrainingTask(cfg, args.alpha ,evaluator)
        #Load the model weights if task is not 0
        if "load_model" in cfg.schedule:
            ckpt = torch.load(cfg.schedule.load_model)
            load_model_weight(TrainTask.model, ckpt, logger)
            load_model_weight(TrainTask.teacher, ckpt, logger)
            logger.info("Loaded model weight from {}".format(cfg.schedule.load_model))
            '''
            for i in range(0, 4):
                init.kaiming_uniform_(TrainTask.model.head.gfl_cls[0].weight[19:20], nonlinearity='relu')
                init.constant_(TrainTask.model.head.gfl_cls[i].bias[19:20], -4.5950)
            '''
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
        max_epochs=cfg.schedule.epochs,
        check_val_every_n_epoch=cfg.schedule.val_interval,
        accelerator=accelerator,
        devices=cfg.device.gpu_ids,
        log_every_n_steps=cfg.log.interval,
        num_sanity_val_steps=0,
        callbacks=[TQDMProgressBar(refresh_rate=0)],# TrainTask.early_stop_callback],
        logger=logger,
        benchmark=cfg.get("cudnn_benchmark", True),
        gradient_clip_val=cfg.get("grad_clip", 0.0),
        strategy=strategy,
        precision=precision,
    )
    trainer.fit(TrainTask, train_dataloader, val_dataloader, ckpt_path=model_resume_path)
    
    'Save only the student model'
    state_dict = TrainTask.model.state_dict()
    new_state_dict = {k: v for k, v in state_dict.items() if "teacher" not in k}
    
    torch.save({'state_dict': new_state_dict}, 'models/task' + str(task) + '/model_last.ckpt')