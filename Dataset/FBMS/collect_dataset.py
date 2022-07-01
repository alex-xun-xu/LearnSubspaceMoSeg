import numpy as np
import tensorflow as tf
import os
import sys
import deepdish
import argparse as arg
import time
import parse
import matplotlib.pyplot as plt

# sys.path.append('/home/elexuxu/vision/Alex/GCN/gcn/gcn')
# sys.path.append(os.path.expanduser('~/vision/Alex/IncompleteData/CorresNet/Model'))
# sys.path.append('/home/elexuxu/vision/Alex/MultiTypeFit/Tools')
# sys.path.append(os.path.expanduser('~/vision/Alex/IncompleteData/Support'))

# sys.path.append(os.path.abspath('../../../GCN/gcn/gcn'))
# sys.path.append(os.path.abspath('../../../CorresNet/Model'))
# sys.path.append(os.path.abspath('../../../../MultiTypeFit/Tools'))
# sys.path.append(os.path.abspath('../../../Support'))

# import SampleCorresNet as model
# import CorresNet_My as model
#
# import MultiTypeFitSupport as supp
# import Losses
# import Tool



def LoadGT(gt_path,seq_name, trg_base_path):
    ###### Load GT
    ## Load dat file
    Mask = []
    Frame = []

    dat_filepath = os.path.join(gt_path, '{:s}Def.dat'.format(seq_name))

    fid = open(dat_filepath, 'r')

    content = fid.readlines()

    for l_i in range(0, len(content)):

        line = content[l_i]

        if 'Total number of frames in this shot' in line:
            seq_len = parse.parse('{:d}\n', content[l_i + 1])[0]

        if 'Total number of labeled frames' in line:
            num_label_frames = parse.parse('{:d}\n', content[l_i + 1])[0]

        if 'File name' in line:
            ## load the mask file/image
            gt_filepath = os.path.join(gt_path,
                                       parse.parse('{}\n', content[l_i + 1])[0])
            frame_no = parse.parse('{:d}\n', content[l_i - 1])[0] + 1
            img = plt.imread(gt_filepath)
            Mask.append(img)
            Frame.append(frame_no)

            os.system('cp {} {}'.format(gt_filepath, trg_base_path))

    fid.close()

    return Mask, Frame, seq_len




parser = arg.ArgumentParser(description='Take parameters')

parser.add_argument('--GPU',type=int,help='GPU to use',default=0)
parser.add_argument('--Depth',type=int,help='Depth of ResNet',
                    default=50)
parser.add_argument('--ExpSum',type=int,help='Flag to indicate if export summary',default=0)    # bool
parser.add_argument('--SaveMdl',type=int,help='Flag to indicate if save learned model',default=0)    # bool
parser.add_argument('--LearningRate',type=float,help='Learning Rate',
                    default=1e-2)
parser.add_argument('--Epoch',type=int,help='Number of epochs to train',default=300)
parser.add_argument('--HiddenDim',type=int,help='Hidden dimension',default=256)
parser.add_argument('--Continue',type=int,help='Flag to indicate whether continue training',default=0)


args = parser.parse_args()
Para = {}
Para['GPU'] = args.GPU
Para['depth'] = args.Depth
Para['SUMMARY_LOG_FLAG'] = args.ExpSum
Para['SAVE_MDL_FLAG'] = args.SaveMdl
Para['LearningRate'] = args.LearningRate
Para['Epoch'] = args.Epoch
HiddenDim = args.HiddenDim
ContinueFlag = args.Continue

if Para['GPU'] != -1:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(Para['GPU'])

## Parameters
MaxClust = 10
output_dim = 1*MaxClust
FixedLen = 10 # fixed length of clip
X_dim = 2*FixedLen
y_dim = MaxClust
gap_str = '1'

lossname = 'L2Loss'

MODEL_NAME = 'CorresNet{:d}_{}'.format(Para['depth'],lossname)
DATA_DIR = os.path.expanduser('./results/')
RESULT_DIR = os.path.join(DATA_DIR, '{}_fixedLen-{}_HiddenDim-{:d}_Results'.
                          format(MODEL_NAME, FixedLen, HiddenDim))
CHECKPOINT_PATH = os.path.join(RESULT_DIR, 'CheckPoint')
SUMMARY_PATH = os.path.join(RESULT_DIR, 'Summary')
summary_filepath = os.path.join(SUMMARY_PATH, 'Summary.txt')
EMBED_PATH = os.path.join(RESULT_DIR,'Embedding')


if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)

if not os.path.exists(CHECKPOINT_PATH):
    os.makedirs(CHECKPOINT_PATH)

if not os.path.exists(SUMMARY_PATH):
    os.makedirs(SUMMARY_PATH)


seq_tr = 'cars2,cars3,cars6,cars7,cars8,cars9,marple1,marple3,marple5,marple8,marple10,marple11,marple13,bear01,bear02,cats02,cats04,cats05,cats07,ducks01,horses01,horses03,horses06,lion02,meerkats01,people04,people05,rabbits01,rabbits05'
seqlist_tr = seq_tr.split(',')

seq_te = 'cars1,cars4,cars5,cars10,marple2,marple4,marple6,marple7,marple9,marple12,people1,people2,tennis,camel01,cats01,cats03,cats06,dogs01,dogs02,farm01,goats01,horses02,horses04,horses05,lion01,people03,giraffes01,rabbits02,rabbits03,rabbits04'
seqlist_te = seq_te.split(',')


# Train on current clip
loss_tr_epoch = 0.

# Random shuffle sequence
seqlist_tr_epoch = seqlist_tr
np.random.shuffle(seqlist_tr_epoch)

for seq_name in seqlist_tr_epoch:

    ## Make Target Directories
    trg_base_path = os.path.join('./Data', seq_name)
    trg_seq_path = os.path.join(trg_base_path,'Seq')

    if not os.path.exists(trg_base_path):
        os.makedirs(trg_base_path)
    if not os.path.exists(trg_seq_path):
        os.makedirs(trg_seq_path)
    ## Directories
    base_path = os.path.expanduser(
        '/vision02/Data/FBMS/moseg/Trainingset/Results/OchsBroxMalik8_all_0000060.00/{}'.format(seq_name))
    track_path = os.path.join(base_path, 'Tracking')
    gt_path = os.path.expanduser('/vision02/Data/FBMS/moseg/Trainingset/{:s}/GroundTruth'.format(seq_name))
    seq_path = os.path.expanduser('/vision02/Data/FBMS/moseg/Trainingset/{:s}'.format(seq_name))
    newclip_path = os.path.join(base_path, 'NewClip')

    ## Copy GT File
    dat_filepath = os.path.join(gt_path, '{:s}Def.dat'.format(seq_name))

    Mask, LabelledFrames, seq_len = LoadGT(gt_path, seq_name, trg_base_path)

    ## Copy BMF file
    bmf_filepath = os.path.join(seq_path, '{:s}.bmf'.format(seq_name))
    os.system('cp {} {}'.format(bmf_filepath, trg_base_path))

    ## Train each clip
    # random shuffle clips
    clips = list(range(0, len(LabelledFrames)))

    os.system('cp {} {}'.format(dat_filepath, trg_base_path))

    ## Frame names
    bmf_filepath = os.path.join(seq_path, '{:s}.bmf'.format(seq_name))
    fid = open(bmf_filepath)
    temp = parse.parse('{:d} {:d}', fid.readline())
    frame_filenames = []
    for f_i in range(0, seq_len):
        line = fid.readline()
        try:
            frame_filenames.append(parse.parse('{}\n', line)[0])
        except:
            frame_filenames.append(line)

    ## Copy all clips
    for clip_i in clips:

        # Load New Clip Data
        newclip_filepath = os.path.join(newclip_path, 'Clip-{:d}_nframe-{:d}_gap-{:s}.h5'.
                                        format(clip_i, FixedLen, gap_str))

        # Data = deepdish.io.load(newclip_filepath)['Data']
        os.system('cp {} {}'.format(newclip_filepath, trg_base_path))


        for f_i in frame_filenames:

            # Visualize original frame
            frame_path = os.path.join(seq_path, '{:s}'.format(f_i))
            os.system('cp {} {}'.format(frame_path, trg_seq_path))

for seq_name in seqlist_te:

    ## Make Target Directories
    trg_base_path = os.path.join('./Data', seq_name)
    trg_seq_path = os.path.join(trg_base_path,'Seq')

    if not os.path.exists(trg_base_path):
        os.makedirs(trg_base_path)
    if not os.path.exists(trg_seq_path):
        os.makedirs(trg_seq_path)

    ## Directories
    base_path = os.path.expanduser('/vision02/Data/FBMS/moseg/Testset/Results/OchsBroxMalik8_all_0000060.00/{}'.format(seq_name))
    track_path = os.path.join(base_path, 'Tracking')
    gt_path = os.path.expanduser('/vision02/Data/FBMS/moseg/Testset/{:s}/GroundTruth'.format(seq_name))
    seq_path = os.path.expanduser('/vision02/Data/FBMS/moseg/Testset/{:s}'.format(seq_name))
    newclip_path = os.path.join(base_path, 'NewClip')

    ## Load GT
    dat_filepath = os.path.join(gt_path, '{:s}Def.dat'.format(seq_name))
    os.system('cp {} {}'.format(dat_filepath, trg_base_path))

    Mask, LabelledFrames, seq_len = LoadGT(gt_path, seq_name, trg_base_path)

    ## Copy BMF file
    bmf_filepath = os.path.join(seq_path, '{:s}.bmf'.format(seq_name))
    os.system('cp {} {}'.format(bmf_filepath, trg_base_path))

    ## Evaluate 1st clip of each testing sequence
    clips = list(range(0, len(LabelledFrames)))

    ## Frame names
    bmf_filepath = os.path.join(seq_path, '{:s}.bmf'.format(seq_name))
    fid = open(bmf_filepath)
    temp = parse.parse('{:d} {:d}', fid.readline())
    frame_filenames = []
    for f_i in range(0, seq_len):
        line = fid.readline()
        try:
            frame_filenames.append(parse.parse('{}\n', line)[0])
        except:
            frame_filenames.append(line)

    for clip_i in clips:

        # Load New Clip Data
        newclip_filepath = os.path.join(newclip_path, 'Clip-{:d}_nframe-{:d}_gap-{:s}.h5'.
                                        format(clip_i, FixedLen, gap_str))

        Data = deepdish.io.load(newclip_filepath)['Data']
        os.system('cp {} {}'.format(newclip_filepath, trg_base_path))

        for f_i in frame_filenames:

            # Visualize original frame
            frame_path = os.path.join(seq_path, '{:s}'.format(f_i))
            os.system('cp {} {}'.format(frame_path, trg_seq_path))