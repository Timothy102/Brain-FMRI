import torch
import os
import sys
import cv2
import shutil
import matplotlib.pyplot as plt

from torch.cuda import init

from Code.utils.initialize import initfolder 
from Code.utils.data_loader import transformation


def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--entry_path", type=str, default= PATH,
                        help="File path to the folder containing the data")
    parser.add_argument("--model_path", type = str, default = SAVED_MODEL_PATH,
                        help = "Folder path containing the model (pth)")
    parser.add_argument("--store_path", type = str, default = VIS_PATH,
                        help = "Set the root path to store the brain segmentations")
    args = parser.parse_args()
    initfolder(args.entry_path), initfolder(args.store_path)
    
    return args

def restore_model(path):
    model = att_autoencoder().to(device)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

def visualize(model, entry_path, store_path):
    f = cv2.imread(entry_path)
    f = transformation(f)
    segmented = model(f)

    plt.plot(segmented)
    plt.show()
    cv2.imwrite(store_path, segmented)
    shutil.rmtree(store_path)


def main(args = sys.argv[1:]):
    args = parseArguments()
    model = restore_model(args.model_path)
    visualize(model, args.entry_path, args.store)
    
if __name__ == '__main__':
    main()