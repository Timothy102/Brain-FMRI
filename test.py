# !/usr/bin/env python3

import os
import shutil
import sys
import cv2

import torch
import torch.functional as F
from torchvision import transforms

from Code.utils import initfolder
from model import att_autoencoder

from Code.utils.data_loader import BrainFMRIDataset, all_paths, ReshapeTensor
from config import INPUT_SHAPE, PATH, SAVED_MODEL_PATH, VIS_PATH

device = torch.device('cude' if torch.cuda.is_available() else 'cpu')


def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default= PATH,
                        help="File path to the folder containing the data")
    parser.add_argument("--model_path", type = str, default = SAVED_MODEL_PATH,
                        help = "Folder path containing the model (pth)")
    parser.add_argument("--vis_path", type = str, default = VIS_PATH,
                        help = "Set the root path to store the brain segmentations")
    args = parser.parse_args()
    initfolder(args.path)
    
    return args

def restore_model(path):
    model = att_autoencoder().to(device)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

def run_inference(model, test_loader, store_path = VIS_PATH,  upper_limit = 100):
    start = 0
    for idx, image, path in enumerate(test_loader):
        image = image.to(device)
        output = model(image)
        
        filepath = path + str(start) + ".jpg"
        initfolder(filepath)
        cv2.imwrite(filepath, output)

        shutil.rmtree(filepath)

        start += 1
        if start > upper_limit:
            break

    print('Testing done.')

def main(args = sys.argv[1:]):
    args = parseArguments()
    test_loader = BrainFMRIDataset(
        images_path = args.path,
        transform = transforms.Compose([
            transforms.ToTensor(),
            ReshapeTensor(INPUT_SHAPE),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225]),
        ]),
        test = True
    )
    model = restore_model(args.model_path)
    run_inference(model, test_loader, store_path = args.store_path, upper_limit = 100)


if __name__ == '__main__':
    main()