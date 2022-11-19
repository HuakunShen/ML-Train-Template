import argparse
from torchvision import datasets, transforms
from src.model.mnist_nets import MnistNet
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from src.utils.config import WandbConfig
from torch.utils.data import Dataset, DataLoader
from src.constant import ROOT
import torch
from src.utils.config import Config
from src.trainer.mnist_trainer import Trainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, help="Learning Rate", default=1.0)
    parser.add_argument("--epochs", type=int, help="Epochs", default=15)
    parser.add_argument("--batch-size", type=int,
                        help="Batch Size", default=64)

    args = parser.parse_args()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    model = MnistNet()
    optimizer = torch.optim.Adadelta(model.parameters(), lr=1)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
    train_dataset = datasets.MNIST(
        ROOT.parent / 'dataset', train=True, download=True, transform=transform)
    # train_dataset = torch.utils.data.Subset(train_dataset, list(range(1000))) # TODO: remove this line

    test_dataset = datasets.MNIST(
        ROOT.parent / 'dataset', train=False, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    workspace = ROOT.parent / 'workspace'
    criterion = F.nll_loss
    wandb_config = WandbConfig("ml-training-template", "huakunai", True)
    config = Config(args.epochs, args.batch_size,
                    1, workspace, 5, wandb_config)

    trainer = Trainer(model, optimizer, scheduler, criterion,
                      config, train_dataset, train_loader, test_loader)
    trainer.train()
