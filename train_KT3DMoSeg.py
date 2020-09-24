import numpy as np
import tensorflow as tf
import os
import sys
import argparse as arg
import time

sys.path.append(os.path.abspath('./Trainer'))

import trainer_KT3DMoSeg as trainer

# import CorresNet_My as model
# import MultiTypeFitSupport as supp
# import KT3DMoSeg_Support as kt_supp
# import Tool
# import Losses
# import MaxInterMinIntra


parser = arg.ArgumentParser(description='Take parameters')

parser.add_argument('--GPU',type=int,help='GPU to use',default=1)
parser.add_argument('--Split',type=int,help='Which cross-validation split to train',
                    default=4)
parser.add_argument('--Depth',type=int,help='Depth of ResNet',
                    default=30)
# parser.add_argument('--ExpSum',type=int,help='Flag to indicate if export summary',default=1)    # bool
parser.add_argument('--SaveMdl',type=int,help='Flag to indicate if save learned model',default=1)    # bool
parser.add_argument('--LearningRate',type=float,help='Learning Rate',
                    default=1e-3)
parser.add_argument('--Epoch',type=int,help='Number of epochs to train',default=200)
parser.add_argument('--HiddenDim',type=int,help='Hidden dimension',default=64)
parser.add_argument('--SavePath',type=str,help='Base saving path',default=None)

args = parser.parse_args()

### Intialize Trainer
Trainer = trainer.Trainer(args)

### Define Network and Solver
Trainer.DefineNetwork()
Trainer.DefineOptimizer()

### Prepare Saving Results & Export Settings
if args.SaveMdl:
    Trainer.PrepareSaveResults(args.SavePath)
    Trainer.SaveAllSettings()


### Start Training
for epoch in range(0,args.Epoch):

    Trainer.TrainOneEpoch()
    print('\nFinished {}-th Epoch\n'.format(epoch))


