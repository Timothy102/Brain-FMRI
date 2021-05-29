import enum
import torch
import os
import sys
import shutil
import argparse

from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss, MSELoss
from torchvision import transforms
import torch.nn.functional as F


from torch.utils.data import DataLoader
from data_loader import BrainFMRIDataset, all_paths, ReshapeTensor
from unet import UNet, NeuralNet
from config import *

            

def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default= PATH,
                        help="File path to the folder containing the data")
    parser.add_argument("--epochs", type=int, default= EPOCHS,
                        help="Select the number of epochs to train")
    parser.add_argument("--batch_size", type=int, default= BATCH_SIZE,
                        help="Select the batch size for training")
    parser.add_argument("--store_path", type=str, default= STORE_PATH,
                        help="Filepath to the folder you wish to store the model")
    parser.add_argument("--model_path", type=str, default= MODEL_CHECKPOINT_PATH,
                        help="Filepath to the folder you wish to store the model checkpoints")
    parser.add_argument("--lr", type = float, default= LEARNING_RATE,
                        help = "Set the learning rate")
    args = parser.parse_args()
    initfolder(args.store_path), initfolder(args.model_path)
    
    return args
    

def joint_loss(pred, mask):
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)
    return (wbce + wiou).mean()

def train(train_loader, epochs, train_save, lr = 1e-2):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NeuralNet()
    model = model.to(device)

    criterion = MSELoss().to(device)
    optimizer = Adam(model.parameters(), lr = lr)
    for epoch in range(epochs):
        running_loss = 0.0
        model.train()
        for idx, (image, mask) in enumerate(train_loader):
            image, mask = image.to(device), mask.to(device)
            optimizer.zero_grad()

            outputs = model(image)
            loss = criterion(outputs, mask)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if idx % 1000 == 999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, idx + 1, running_loss / 2000))
                torch.save(model.state_dict(),
                './Snapshots/save_weights/{}/unet_model_{}.pkl'.format(train_save, idx+1))
                print("Saving model checkpoint: model_{}.pkl".format(idx+1))
            running_loss = 0.0


def main(args = sys.argv[1:]):
    args = parseArguments()
    t = transforms.Compose([
        transforms.ToTensor(),
        ReshapeTensor((3, 256, 256))
    ])
    images_path, masks_path = all_paths(args.path)
    dataset = BrainFMRIDataset(images_path, masks_path, transform = t)
    data_loader = DataLoader(dataset, args.batch_size, shuffle = True)
    train(data_loader, args.epochs, args.store_path, lr = args.lr)


if __name__ == "__main__":
    main()