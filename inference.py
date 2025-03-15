import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import pickle
from tqdm import tqdm

from models.densenet_121_27 import densenet_cifar
from utils.customDataset import customDataset

import logging
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mean = [0.49139967861519607,0.48215840839460783,0.44653091444546567]
std = [0.24703223246328176,0.24348512800005648,0.26158784172796473]

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
    cifar10_batch = load_cifar_batch('/scratch/ky2684/dl-project-1/cifar-classification/data/cifar_test_nolabel.pkl')

    # Extract images 
    test_images = cifar10_batch[b'data']

    checkpoint = torch.load('/scratch/ky2684/dl-project-1/cifar-classification/ckpts/best_model.pth')
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
    test_model_fn()