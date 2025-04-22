import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import pickle
from tqdm import tqdm
import os

from models.densenet_121_27 import densenet_cifar
from utils.customDataset import customDataset

import logging
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mean = [0.49139967861519607,0.48215840839460783,0.44653091444546567]
std = [0.24703223246328176,0.24348512800005648,0.26158784172796473]
num_workers = os.cpu_count()

def load_cifar_batch(file):
    with open(file, 'rb') as fo:
        batch = pickle.load(fo, encoding='bytes')
    return batch


def validate(model, val_loader, criterion, dataset_size):
    model.eval()
    total_loss, correct = 0.0, 0
    
    with torch.no_grad():
        for inputs, targets in tqdm(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total_loss += loss.item() * inputs.size(0)
    
    acc = 100 * correct / dataset_size
    
    print(f"Validation Accuracy: {acc:.2f}%")

def validate_model_fn(batch_size=128, ):
    # Load the batch
    cifar10_dir = '/scratch/ky2684/CSGY-6953-deep-learning/data/cifar-10-python/cifar-10-batches-py'
    
    val_dict = load_cifar_batch(os.path.join(cifar10_dir, 'test_batch'))
    val_images = val_dict[b'data']
    val_labels = val_dict[b'labels']

    val_images = val_images.reshape((10000, 3, 32, 32)).transpose(0, 2, 3, 1)

    val_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    val_dataset = customDataset(val_images, val_labels, val_transforms)

    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, 
                                pin_memory=True, persistent_workers=True, prefetch_factor=2)

    val_dataset_size = len(val_dataset)

    checkpoint = torch.load('/scratch/ky2684/CSGY-6953-deep-learning/ckpts/best_model_train_densenet27_600.pth')
    model = densenet_cifar().to(device)
    model = nn.DataParallel(model)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    criterion = nn.CrossEntropyLoss()

    validate(model, val_loader=val_dataloader, criterion=criterion, dataset_size=val_dataset_size)


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
    results_df.to_csv('submission_final_test.csv', index=False)
    
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
    # validate_model_fn()
    test_model_fn()