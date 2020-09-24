import os
import sys
import argparse as arg
import pathlib
import datetime


parser = arg.ArgumentParser(description='Take parameters')

parser.add_argument('--GPU',type=int,help='GPU to use',default=2)
parser.add_argument('--Epoch',type=int,help='GPU to use',default=300)
parser.add_argument('--ExpFig',type=int,help='Flag to indicate if export figure',default=0)  # bool
# parser.add_argument('--ExpSum',type=int,help='Flag to indicate if export summary',default=1)    # bool
parser.add_argument('--SaveMdl',type=int,help='Flag to indicate if save learned model',default=1)    # bool
parser.add_argument('--ContinueTrain',type=int,help='Flag to indicate if continue training from existing checkpoint',
                    default=0)
parser.add_argument('--Depth',type=int,help='ResNet Depth',
                    default=30)
parser.add_argument('--SeqLen',type=int,help='The fixed Length of sequence to use',
                    default=5)
parser.add_argument('--LearningRate',type=float,help='Learning Rate',
                    default=1e-3)
parser.add_argument('--EmbedDim',type=int,help='Embed Dim',
                    default=5)
parser.add_argument('--HiddenDim',type=int,help='Hidden dimension',default=64)

args = parser.parse_args()
Setting = {}
Setting['GPU'] = args.GPU
Setting['EXPORT_FIG_FLAG'] = bool(args.ExpFig)
Setting['SAVE_MDL_FLAG'] = bool(args.SaveMdl)
Setting['num_epoch'] = args.Epoch
CONTINUE_TRAIN_FLAG = args.ContinueTrain
Setting['depth'] = args.Depth
FixedLen = args.SeqLen
Setting['LearningRate'] = args.LearningRate # ResNet Depth
Setting['Dataset'] = 'KT3DMoSeg' # dataset
EmbedDim = args.EmbedDim # ResNet Depth
HiddenDim = args.HiddenDim

### Paramters
nCV = 22
NumInstance = 22
# FixedLen = 5
MaxClust = 5
# task = 'Fundamental'
lossname = 'MaxInterMinIntra'

split_range = list(range(0,nCV))

### Prepare Results Saving Path
base_path = os.path.abspath(os.path.join(str(pathlib.Path(__file__).parent.absolute()),'./Results'))
Setting['result_base_path'] = os.path.join(base_path, Setting['Dataset'])
if not os.path.exists(Setting['result_base_path']):
    os.makedirs(Setting['result_base_path'])
time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # get current time


# Run cross-validation

for split in split_range:

    print('Split - {:d} starts\n'.format(split))
    # fid = open(summary_filepath, 'a+')
    # fid.close()

    ### Create Spit Save Directory
    Setting['result_path'] = os.path.join(Setting['result_base_path'], '{}'.
                                          format(time), 'split-{}'.format(split))  # complete the result path
    os.makedirs(Setting['result_path'])
    # summary output path
    Setting['summary_path'] = os.path.join(Setting['result_path'], 'summary')

    command = 'python ./train_KT3DMoSeg.py ' \
              '--Depth {:d} --GPU {:d} --Split {:d} ' \
              '--Epoch {:d} ' \
              '--LearningRate {:f} --SaveMdl {:d} --HiddenDim {:d} --SavePath {}'. \
        format(Setting['depth'], Setting['GPU'], split, Setting['num_epoch'],
               Setting['LearningRate'], Setting['SAVE_MDL_FLAG'], HiddenDim, Setting['result_path'])

    os.system(command)


    # Check if computed
    # MODEL_NAME = 'ResNet{:d}_{}'.format(ResNetDepth, lossname)
    # MODEL_NAME = 'SubspaceNetShrink{:d}_{}'.format(Para['depth'], lossname)
    # DATA_DIR = os.path.expanduser('~/vision/Alex/Data/MultiTypeFit/CorresNet/KT3DMoSeg')
    # RESULT_DIR = os.path.join(DATA_DIR,
    #                           '{}_fixedLen-{}_nCV-{:d}_HiddenDim-{:d}_Results'.format(MODEL_NAME, FixedLen, nCV,
    #                                                                                   HiddenDim))
    # CHECKPOINT_PATH = os.path.join(RESULT_DIR, 'CheckPoint')
    # SUMMARY_PATH = os.path.join(RESULT_DIR, 'Summary')
    # summary_filepath = os.path.join(SUMMARY_PATH, 'Summary_tesplit-{:d}.txt'.format(split))
    # EMBED_PATH = os.path.join(RESULT_DIR, 'Embedding')

    # if not os.path.exists(SUMMARY_PATH):
    #     os.makedirs(SUMMARY_PATH)
    #
    # if os.path.isfile(summary_filepath):
    #     print('Split - {:d} exists\n'.format(split))
    #     continue
    # else:
    #     print('Split - {:d} starts\n'.format(split))
    #     fid = open(summary_filepath,'a+')
    #     fid.close()
    #
    #     command = 'python ./train_SubspaceNet_Shrink_{}.py ' \
    #               '--Depth {:d} --GPU {:d} --Split {:d} ' \
    #               '--Epoch {:d} --ExpSum {:d} ' \
    #               '--LearningRate {:f} --SaveMdl {:d} --HiddenDim {:d}'.\
    #         format(lossname, Para['depth'],Para['GPU'],split,Para['num_epoch'],
    #                Para['SUMMARY_LOG_FLAG'],Para['LearningRate'],Para['SAVE_MDL_FLAG'],HiddenDim)
    #
    #     os.system(command)