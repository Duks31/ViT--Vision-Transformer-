import matplotlib.pyplot as plt
import torch
import torchvision

import utils
import engine

from torch import nn
from torchvision import transforms
from pathlib import Path
from torchinfo import summary
from vit import ViT


data_path = Path("data")
BATCH_SIZE = 32 
IMG_SIZE = 224
device = "cuda" if torch.cuda.is_available() else "cpu"

train_dir = data_path / "train"
test_dir = data_path / "test"

manual_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])           

train_dataloader, test_dataloader, class_names = utils.create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    transform=manual_transforms, # use manually created transforms
    batch_size=BATCH_SIZE
)

vit = ViT(num_classes=len(class_names))

optimizer = torch.optim.Adam(
    params=vit.parameters(),
    lr = 3e-3,
    betas = (0.9, 0.999),
    weight_decay=0.3,
)

loss_fn = nn.CrossEntropyLoss()

if __name__ == "__main__":
    results = engine.train(
    model = vit,
    train_dataloader= train_dataloader,
    test_dataloader= test_dataloader,
    optimizer=optimizer,
    loss_fn=loss_fn,
    epochs = 10,
    device = device
    )