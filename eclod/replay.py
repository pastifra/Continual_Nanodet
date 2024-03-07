import argparse
import os
import warnings
import yaml

import sys
sys.path.append('/home/pasti/PycharmProjects/Continual_Nanodet/') #Change this to the path of the cloned repository

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
import random
from torch.utils.data import Subset
from IPython.display import Image
from IPython.display import display
from nanodet.data.dataset.coco import CocoDataset

class ReplayDataLoader:
    def __init__(self, dataset1, dataset2, batch_size, shuffle):
        """
        This class is used to create a dataloader that creates batches
        using the task specific dataset and the replay buffer dataset.
        Batches are created using 50% of the task specific dataset and 50% of the replay buffer dataset.
        The iterators are reset when the task specific dataset is exhausted.
        If the replay buffer dataset is exhausted before the task specific dataset, its iterator is reset.
        
        Args:
            dataset1: task specific dataset
            dataset2: replay buffer dataset
            batch_size: Batch size
            shuffle: Whether to shuffle the data
        """
        
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.dataset1_loader = torch.utils.data.DataLoader(
            self.dataset1,
            batch_size=self.batch_size//2,
            shuffle=self.shuffle,
            num_workers=8,
            pin_memory=True,
            collate_fn=naive_collate,
            drop_last=True,
        )
        self.dataset2_loader = torch.utils.data.DataLoader(
            self.dataset2,
            batch_size=self.batch_size//2,
            shuffle=self.shuffle,
            num_workers=8,
            pin_memory=True,
            collate_fn=naive_collate,
            drop_last=True,
        )
    
    def __iter__(self):
        self.dataset1_iter = iter(self.dataset1_loader)
        self.dataset2_iter = iter(self.dataset2_loader)
        return self
    
    def __next__(self):
        try:
            batch1 = next(self.dataset1_iter)
        except StopIteration:
            raise StopIteration
        try:
            batch2 = next(self.dataset2_iter)
        except StopIteration:
            self.dataset2_iter = iter(self.dataset2_loader)
            batch2 = next(self.dataset2_iter)
            
        merged_batch = {}
        for key in batch1.keys():
            if key == 'img':
                merged_batch[key] = batch1[key] + batch2[key]
            elif key == 'img_info':
                merged_batch[key] = {k: batch1[key][k] + batch2[key][k] for k in batch1[key]}
            elif key in ['gt_bboxes', 'gt_labels', 'gt_bboxes_ignore', 'warp_matrix']:
                #merged_batch[key] = [torch.cat((torch.tensor(b1), torch.tensor(b2))) for b1, b2 in zip(batch1[key], batch2[key])]
                merged_batch[key] = batch1[key] + batch2[key]
            else:
                raise ValueError(f"Key not recognized")
        
        return merged_batch
    
    def __len__(self):
        return len(self.dataset1)
        
class StandardBufferDataset(Dataset):
    """
    This class is used to create a replay buffer dataset that stores random samples of the task specific dataset.
    At init it takes random samples of the first task dataset to fill the buffer.
    Then from task n to task n+1 it, where n>0, it updates 50% of the buffer with the new task dataset.
    
    Args:
        dataset_n: task specific dataset
        buffer_size: size of the replay buffer
    """
    def __init__(self, dataset_n, buffer_size=250):
        self.buffer_size = buffer_size
        #At initialization, take random samples of the task 0 dataset to fill the buffer
        self.buffer_indices = random.sample(range(len(dataset_n)), self.buffer_size)
        self.buffer_dataset = Subset(dataset_n, self.buffer_indices)

    def __getitem__(self, buff_index):
        #Just return a buffer item at the index
        return self.buffer_dataset[buff_index]

    def __len__(self):
        #Return the buffer size
        return self.buffer_size

    def update_buffer(self, dataset_np1):
        #Take a random subset of the old buffer
        if len(dataset_np1) < int(self.buffer_size/2):
            update_buffer_indices = random.sample(range(self.buffer_size), self.buffer_size - len(dataset_np1))
            subset_n = Subset(self.buffer_dataset, update_buffer_indices)
            
            self.buffer_dataset = torch.utils.data.ConcatDataset([subset_n, dataset_np1])        
        else:
            update_buffer_indices = random.sample(range(self.buffer_size), int(self.buffer_size/2))
            subset_n = Subset(self.buffer_dataset, update_buffer_indices)
    
            #Take a random subset of the new task dataset
            new_rand_indices = random.sample(range(len(dataset_np1)), int(self.buffer_size/2))
            subset_np1 = Subset(dataset_np1, new_rand_indices)

            #Concate the two subsets to form the new buffer
            self.buffer_dataset = torch.utils.data.ConcatDataset([subset_n, subset_np1])

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

'''
####TEMP INITIALIZATION of buffer when the base model is already available
task = 0
create_exp_cfg('cfg/VOC.yml', task, CLtask)
load_config(cfg, 'cfg/' + CLtask + 'task' + str(task) + '.yml')
#Build datasets and dataloaders based on the task configuration file
train_dataset = build_dataset(cfg.data.train, "train")
#val_dataset = build_dataset(cfg.data.val, "test")
buffer_dataset = StandardBufferDataset(train_dataset)
'''
#Set logger and seed
logger = NanoDetLightningLogger('test')
pl.seed_everything(1234)

parser = argparse.ArgumentParser(description="Parser for training task")
parser.add_argument('--task', type=str, help='Type of CL task to train (15p5, 10p10, 19p1, 15p1)', required=True)
parser.add_argument('--cfg', type=str, help='Path to the configuration file', required=True)

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
    if task == 0:
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=cfg.device.batchsize_per_gpu,
            shuffle=True if task == 0 else False, #Shuffling is done inside ReplayDataset class
            num_workers=cfg.device.workers_per_gpu,
            pin_memory=True,
            collate_fn=naive_collate,
            drop_last=True,
        )
    else:
        train_dataloader = ReplayDataLoader(
            train_dataset, 
            buffer_dataset, 
            cfg.device.batchsize_per_gpu,
            shuffle = True
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
    logger.info("Creating model...")
    TrainTask = TrainingTask(cfg, evaluator)
    #Load the model weights if task is not 0
    if "load_model" in cfg.schedule:
        ckpt = torch.load(cfg.schedule.load_model)
        if "pytorch-lightning_version" not in ckpt:
            warnings.warn(
                "Warning! Old .pth checkpoint is deprecated. "
                "Convert the checkpoint with tools/convert_old_checkpoint.py "
            )
            ckpt = convert_old_model(ckpt)
        load_model_weight(TrainTask.model, ckpt, logger)
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
        devices=cfg.device.gpu_ids,
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
    
    #Replay code
    #If task is 0, initialize the replay buffer with the task 0 dataset 
    #if task > 0 update the buffer with the new task dataset
    if task == 0:
        print("Creating buffer dataset")
        buffer_dataset = StandardBufferDataset(train_dataset)
    else:
        print("Updating buffer dataset")
        buffer_dataset.update_buffer(train_dataset)