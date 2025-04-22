import os
import pickle

import numpy as np
import pandas as pd
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.amp import autocast, GradScaler
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from torch.utils.data import default_collate
from torch import optim

from tqdm import tqdm

from utils.customDataset import customDataset
from models.densenet_121_27 import densenet_cifar

import logging
logger = logging.getLogger(__name__)

#---------- FLAGS TO SPEED UP TRAINING ----------#
torch.set_float32_matmul_precision('high')
torch.manual_seed(0)

#---------- WANDB LOGIN ----------#
import wandb
secret_value_0 = "135f8ae5275a37477206fcfc791e6dd933717d21"
wandb.login(key=secret_value_0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#---------- LOAD DATA ----------#
def load_cifar_batch(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# change path
cifar10_dir = '/scratch/ky2684/dl-project-1/cifar-classification/data/cifar-10-python/cifar-10-batches-py'

meta_data_dict = load_cifar_batch(os.path.join(cifar10_dir, 'batches.meta'))
label_names = meta_data_dict[b'label_names']

train_images = []
train_labels = []

for i in range(5):
    batch_dict = load_cifar_batch(os.path.join(cifar10_dir, 'data_batch_' + str(i+1)))
    train_images.append(batch_dict[b'data'])
    train_labels.extend(batch_dict[b'labels'])

train_images = np.concatenate(train_images, axis=0)
train_labels = np.array(train_labels)
train_images = train_images.reshape((50000, 3, 32, 32)).transpose(0, 2, 3, 1)

val_dict = load_cifar_batch(os.path.join(cifar10_dir, 'test_batch'))
val_images = val_dict[b'data']
val_labels = val_dict[b'labels']

val_images = val_images.reshape((10000, 3, 32, 32)).transpose(0, 2, 3, 1)

# Combine training and validation datasets
combined_images = np.concatenate([train_images, val_images], axis=0)
combined_labels = np.concatenate([train_labels, val_labels], axis=0)
    
#---------- DATALOADERS ----------#
# Calculate mean and std over the combined dataset
mean = np.mean(combined_images, axis=(0, 1, 2)) / 255
std = np.std(combined_images, axis=(0, 1, 2)) / 255

def get_dataloaders(batch_size=128, num_workers=2):
    train_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(32, padding=4),
        transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0)
    ])

    # Create the combined dataset with the full training data
    combined_dataset = customDataset(combined_images, combined_labels, train_transforms)
    
    # Create dataloader for the combined dataset
    combined_dataloader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, 
                                  pin_memory=True, persistent_workers=True, prefetch_factor=2)
    
    return combined_dataloader, len(combined_dataset)

#---------- LOAD MODEL ----------#
def setup_model():
    model = densenet_cifar().to(device)
    model = nn.DataParallel(model)
    nn.Module.compile(model)
    return model

#---------- TRAINING FUNCTION FOR ONE EPOCH ----------#

def train_one_epoch(model, train_loader, criterion, optimizer, scaler, epoch, max_grad_norm=1.0):
    model.train()
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False)
    running_loss = 0.0
    correct = 0
    total = 0
    batch_gradient_norms = []
    
    for inputs, targets in loop:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        
        with autocast("cuda"):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        
        scaler.scale(loss).backward()
        
        # Apply gradient clipping before optimizer step
        # When using mixed precision, we need to unscale the gradients first
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        # Calculate gradient norm
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        batch_gradient_norms.append(total_norm)
        
        scaler.step(optimizer)
        scaler.update()
        
        # Compute batch accuracy
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        running_loss += loss.item()
        loop.set_postfix(loss=loss.item())
    
    avg_loss = running_loss / len(train_loader)
    train_acc = 100 * correct / total
    avg_gradient_norm = sum(batch_gradient_norms) / len(batch_gradient_norms)

    # Collect all metrics for epoch-level logging
    epoch_metrics = {
        "train_loss": avg_loss,
        "train_acc": train_acc,
        "epoch": epoch,
        "learning_rate": optimizer.param_groups[0]["lr"],
        "gradient_norm": avg_gradient_norm,
    }
    
    print(f"Epoch {epoch+1} - Training Accuracy: {train_acc:.2f}%")
    
    return epoch_metrics

#---------- MAIN TRAINING FUNCTION ----------# 
 
def train_model(epochs=200, lr=0.1, weight_decay=5e-4, max_grad_norm=1.0, batch_size=128, num_workers=4, beta1=0.9, beta2=0.999, run_name=None):
    # Prepare comprehensive config
    config = {
        # Model parameters
        "model": "DenseNet_final_final",
        "num_classes": 10,
        "dropout_rate": None,
        
        # Training hyperparameters
        "epochs": epochs,
        "learning_rate": lr,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "max_grad_norm": max_grad_norm,
        
        # Optimizer config
        "optimizer": "SGD",
        
        # Scheduler config
        "scheduler": "CosineAnnealingLR",
        "scheduler_params": {
            "T_max": epochs,
        },
        
        # Loss function
        "loss_function": "CrossEntropyLoss",
        
        # Training techniques
        "mixed_precision": True,
        "data_parallel": True,
        
        # Data augmentation
        "data_augmentation": {
            "normalization_mean": mean.tolist(),
            "normalization_std": std.tolist(),
            "auto_augment": True,
            "random_erasing": True,
        },
        
        # Hardware info
        "device": str(device),
        
        # Dataset info
        "dataset": "CIFAR10 (combined train+val)"
    }
    
    # Initialize wandb with all configuration
    wandb.init(
        project="dl-s-25", 
        config=config,
        name=run_name if run_name else "densenet27_combined_datasets",
        tags=["densenet27", "cifar10", "combined_datasets"],
        notes="Training with combined train+val datasets and tracking best train accuracy"
    )
    
    train_loader, dataset_size = get_dataloaders(batch_size=batch_size, num_workers=num_workers)
    model = setup_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-4)
    scaler = GradScaler()
    
    # Create table for tracking hyperparameters over epochs
    columns = ["epoch", "learning_rate", "train_loss", "train_acc", "gradient_norm"]
    epoch_table = wandb.Table(columns=columns)
    
    best_train_acc = 0.0
    for epoch in range(epochs):
        # Train for one epoch and get metrics
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, scaler, epoch, max_grad_norm)
        
        # Step the scheduler
        scheduler.step()
        
        # Log all metrics at once (only once per epoch)
        wandb.log(train_metrics)
        
        # Add row to epoch table
        epoch_table.add_data(
            epoch, 
            optimizer.param_groups[0]["lr"],
            train_metrics["train_loss"],
            train_metrics["train_acc"],
            train_metrics["gradient_norm"]
        )
        
        # Save model checkpoint if it's the best train accuracy so far
        if train_metrics["train_acc"] > best_train_acc:
            checkpoint_path = f'/scratch/ky2684/CSGY-6953-deep-learning/ckpts/best_model.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'accuracy': train_metrics["train_acc"],
                'train_loss': train_metrics["train_loss"],
            }, checkpoint_path)
            
            # Log checkpoint to wandb
            wandb.save(checkpoint_path)
            
            best_train_acc = train_metrics["train_acc"]
            logger.info(f"Best model updated with accuracy: {best_train_acc:.2f}%")
            wandb.run.summary["best_acc"] = best_train_acc
            wandb.run.summary["best_epoch"] = epoch
            
    # Log final epoch table
    wandb.log({"epoch_metrics_table": epoch_table})
    
    # Log final model architecture as artifact
    model_artifact = wandb.Artifact(
        name=f"model-{wandb.run.id}", 
        type="model",
        description="Final trained model"
    )
    model_artifact.add_file("/scratch/ky2684/CSGY-6953-deep-learning/ckpts/best_model.pth")
    wandb.log_artifact(model_artifact)
    
    # Finish the run
    wandb.finish()
    
    return best_train_acc

def run_experiments():
    # Define different configurations to try
    experiments = [
        {"run_name": "github_densenet27_da_final_final", "epochs": 600, "batch_size": 128},
    ]
    
    for exp in experiments:
        print(f"Running experiment: {exp['run_name']}")
        train_model(**exp)

def load_cifar_batch(file):
    with open(file, 'rb') as fo:
        batch = pickle.load(fo, encoding='bytes')
    return batch

def test_model(model, test_loader, device):
    model.eval()
    predictions = []
    ids = []

    with torch.no_grad():
        for inputs, targets in tqdm(test_loader):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            predictions.extend(predicted.cpu().numpy())
            ids.extend(range(len(predictions) - len(predicted), len(predictions)))

    # Create a DataFrame with IDs and predictions
    results_df = pd.DataFrame({'ID': ids, 'Labels': predictions})
    
    # Save the results to a CSV file
    results_df.to_csv('submission_final.csv', index=False)
    
    logger.info("Predictions saved to submission.csv")

def test_model_fn():
    # Load the batch
    cifar10_batch = load_cifar_batch('/scratch/ky2684/CSGY-6953-deep-learning/data/cifar_test_nolabel.pkl')

    # Extract images 
    test_images = cifar10_batch[b'data']

    checkpoint = torch.load('/scratch/ky2684/CSGY-6953-deep-learning/ckpts/best_model.pth')
    model = densenet_cifar().to(device)
    model = nn.DataParallel(model)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    test_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ColorJitter(contrast=(0.95, 0.98)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    test_dataset = customDataset(test_images, transform=test_transforms, is_test = True)
    test_dataloader = DataLoader(test_dataset, batch_size=1024, shuffle=False, num_workers=4)

    test_model(model, test_dataloader, device)

if __name__ == "__main__":
    run_experiments()
    test_model_fn()