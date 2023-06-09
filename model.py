import torch
from torchvision.models import resnet34
import torch.nn as nn

from dataset import get_dataset_loaders

if torch.cuda.is_available():
  device=torch.device('cuda:0')
else:
  device=torch.device('cpu')

train_loader, test_loader, train_dataset, test_dataset = get_dataset_loaders(limit=1000)

resnet_model = resnet34(pretrained=True)
resnet_model.fc = nn.Linear(512,50)
resnet_model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
resnet_model = resnet_model.to(device)

