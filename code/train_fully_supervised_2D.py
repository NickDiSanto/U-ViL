import argparse
import logging
import os
import random
import shutil
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm


from dataloaders import utils
from dataloaders.dataset import BaseDataSets, RandomGenerator, BaseDataSets_Synapse
from networks.net_factory import net_factory
from utils import losses, metrics, ramps
from val_2D import test_single_volume, test_single_volume_ds



parser = argparse.ArgumentParser()
parser.add_argument("--root_path", type=str,
                     default="../home/khodape/Desktop/Datasets/ACDC/ACDC/", help="Name of the experiment")
parser.add_argument("--exp", type=str,
                     default="ACDC/Fully_Supervised", help="experiment_name")
parser.add_argument("--model", type=str, 
                    default="unet", help="model_name")
parser.add_argument("--num_classes", type=int,
                    default=2, help="output channel of net")
parser.add_argument("--max_iterations", type=int,
                    default=10000, help="maximum epoch number to train")
parser.add_argument("--batch_size", type=int, default=24,
                    help="batch_size per gpu")
parser.add_argument("--deterministic", type=int,  default=1,
                    help="whether use deterministic training")
parser.add_argument("--base_lr", type=float,  default=0.01,
                    help="segmentation network learning rate")
parser.add_argument("--patch_size", type=list,  default=[256, 256],
                    help="patch size of network input")
parser.add_argument("--seed", type=int,  default=1337, help="random seed")
parser.add_argument("--labeled_num", type=int, default=140,
                    help="labeled data")
args = parser.parse_args()

def patients_to_slices(dataset, patiens_num):
    ref_dict = None
    if "ACDC" in dataset:
        ref_dict = {"3": 68, "7": 136,
                    "14": 256, "21": 396, "28": 512, "35": 664, "140": 1312}
    else:
        print("Error")
    return ref_dict[str(patiens_num)]



def train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations

    labeled_slice = patients_to_slices(args.root_path, args.labeled_num)
    # print(labeled_slice)
    model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes)
    print(model)
    print(args.model)




if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = "./results/model/{}_{}_labeled/{}".format(args.exp, 
                                                      args.labeled_num, 
                                                      args.model)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    # if os.path.exists(snapshot_path):
    #     shutil.rmtree(snapshot_path)
    # shutil.copytree(".", snapshot_path + "/code",
    #                 shutil.ignore_patterns([".git", "__pycache__"]))

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format="[%(asctime)s.%(msecs)03d] %(message)s", datefmt="%H:%M:%S")
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)

