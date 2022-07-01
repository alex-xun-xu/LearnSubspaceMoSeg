import numpy as np
import tensorflow as tf
import os
import sys
import deepdish
import argparse as arg
import time
import parse
import matplotlib.pyplot as plt
import scipy.io as scio


# sys.path.append('/home/elexuxu/vision/Alex/GCN/gcn/gcn')
# sys.path.append(os.path.expanduser('~/vision/Alex/IncompleteData/CorresNet/Model'))
# sys.path.append('/home/elexuxu/vision/Alex/MultiTypeFit/Tools')
# sys.path.append(os.path.expanduser('~/vision/Alex/IncompleteData/Support'))
# sys.path.append('../../Model')
# sys.path.append('../Export')

sys.path.append(os.path.abspath('../Network'))
sys.path.append(os.path.abspath('../Util'))

import SubspaceNet as model
import Loss
import Tool
import MdlSel
import FBMS_IO as FIO

def LoadGT(gt_path,seq_name):
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

    fid.close()

    return Mask, Frame, seq_len




parser = arg.ArgumentParser(description='Take parameters')

parser.add_argument('--GPU',type=int,help='GPU to use',default=1)
parser.add_argument('--Split',type=int,help='Which cross-validation split to train',
                    default=4)
parser.add_argument('--Depth',type=int,help='Depth of ResNet',
                    default=50)
parser.add_argument('--ExpSum',type=int,help='Flag to indicate if export summary',default=0)    # bool
parser.add_argument('--SaveMdl',type=int,help='Flag to indicate if save learned model',default=0)    # bool
parser.add_argument('--LearningRate',type=float,help='Learning Rate',
                    default=1e-3)
parser.add_argument('--Epoch',type=int,help='Number of epochs to train',default=200)
parser.add_argument('--HiddenDim',type=int,help='Hidden dimension',default=256)
# parser.add_argument('--RsltPath',type=str,help='Result path ../Results/FBMS/XXX (Replace XXX with the result directory)',default='../Results/FBMS/XXX')
parser.add_argument('--RsltPath',type=str,help='Result path ../Results/FBMS/XXX (Replace XXX with the result directory)',default='/home/xuxun/Dropbox/GitHub/LearnSubspaceMoSeg/Results/FBMS/SubspaceNet50_L2Loss_fixedLen-10_HiddenDim-256_2022-07-01_17-35-17')


args = parser.parse_args()
Para = {}
Para['GPU'] = args.GPU
Para['split'] = args.Split
Para['depth'] = args.Depth
Para['SUMMARY_LOG_FLAG'] = args.ExpSum
Para['SAVE_MDL_FLAG'] = args.SaveMdl
Para['LearningRate'] = args.LearningRate
Para['Epoch'] = args.Epoch
HiddenDim = args.HiddenDim

if Para['GPU'] != -1:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(Para['GPU'])

## Parameters
MaxClust = 10
output_dim = 1*MaxClust
FixedLen = 10
X_dim = 2*FixedLen
y_dim = MaxClust
nCV = 22    # number of cross validation folds
N = FixedLen   # fixed length of clip
gap_str = '1'

lossname = 'L2Loss'

MODEL_NAME = 'SubspaceNet{:d}_{}'.format(Para['depth'],lossname)
DATA_DIR = os.path.expanduser('../Results/FBMS')
RESULT_DIR = args.RsltPath
CHECKPOINT_PATH = os.path.join(RESULT_DIR, 'CheckPoint')
SUMMARY_PATH = os.path.join(RESULT_DIR, 'Summary')
summary_filepath = os.path.join(SUMMARY_PATH, 'Summary.txt')
EMBED_PATH = os.path.join(RESULT_DIR,'Embedding')
PRED_PATH = os.path.join(RESULT_DIR,'Prediction_MdlSel_SOD')
MOSEG_PATH = os.path.join(RESULT_DIR,'MoSeg')

if not os.path.exists(PRED_PATH):
    os.makedirs(PRED_PATH)

if not os.path.exists(MOSEG_PATH):
    os.makedirs(MOSEG_PATH)


seq_tr = 'cars2,cars3,cars6,cars7,cars8,cars9,marple1,marple3,marple5,marple8,marple10,marple11,marple13,bear01,bear02,cats02,cats04,cats05,cats07,ducks01,horses01,horses03,horses06,lion02,meerkats01,people04,people05,rabbits01,rabbits05'
seqlist_tr = seq_tr.split(',')

seq_te = 'cars1,cars4,cars5,cars10,marple2,marple4,marple6,marple7,marple9,marple12,people1,people2,tennis,camel01,cats01,cats03,cats06,dogs01,dogs02,farm01,goats01,horses02,horses04,horses05,lion01,people03,giraffes01,rabbits02,rabbits03,rabbits04'
seqlist_te = seq_te.split(',')



## Define placeholder/inputs
# H Feat Embedding B*N*D
H_ph = tf.placeholder(dtype=tf.float32,shape=[None, None, X_dim],name='inputs/input_feature_embedding')
# y groun-truth labels B*N*C
y_gt_ph = tf.placeholder(dtype=tf.float32,shape=[None, None, y_dim],name='inputs/categorical_label')

Is_Training_ph = tf.placeholder(dtype=tf.bool,shape=[])

## Define GCN network
y_hat = model.SubspaceNet(
    H=H_ph,input_dim=X_dim,output_dim=output_dim,Is_Training=Is_Training_ph,hidden_dim=HiddenDim,depth=Para['depth'])

y_hat = tf.nn.l2_normalize(y_hat, axis=2)

loss = Loss.Loss_L2(y_gt_ph,y_hat)

### Define solver
train_vars = tf.trainable_variables()

## Initialize
saver = tf.train.Saver(max_to_keep=2)

config = tf.ConfigProto(allow_soft_placement=False)
config.gpu_options.allow_growth = bool(True)  # Use how much GPU memory

sess = tf.Session(config=config)

## Load Trained Model
max_epoch = 10000

for epoch in range(max_epoch,0,-1):
    mdl_filepath = os.path.join(CHECKPOINT_PATH,
                   "model_epoch-{:s}.ckpt.meta".format('best'))
    if os.path.exists(mdl_filepath):

        saver.restore(sess,os.path.join(CHECKPOINT_PATH,
                   "model_epoch-{:s}.ckpt".format('best')))
        print('Succeeded to restore model - {}'.format(
            "model_epoch-{:s}.ckpt.meta".format('best')))
        break


### Start Testing

## Evaluate each testing example one by one

acc_te_all = []
pred_te_all = []
loss_te_all = 0
acc_te_seq = []

for seq_name in seqlist_te:
    ## Directories
    base_path = os.path.abspath(
        '../Dataset/FBMS/Data/{}'.format(seq_name))
    gt_path = os.path.abspath('../Dataset/FBMS/Data/{:s}'.format(seq_name))
    seq_path = os.path.abspath('../Dataset/FBMS/Data/{:s}/Seq'.format(seq_name))
    newclip_path = base_path

    ## Load GT
    Mask, Frame, seq_len = LoadGT(gt_path, seq_name)

    ## Evaluate each clip
    acc_clip = []

    # for clip_i in range(0, len(Frame)):
    for clip_i in range(0, 1):

        with tf.device('/cpu:0'):

            # Results Saving
            newclip_resultseq_path = os.path.join(RESULT_DIR, 'MoSeg_MdlSel_SOD', seq_name,
                            'Clip-{:d}_nframe-{:d}_gap-{:s}'.format(clip_i, N, gap_str))

            if not os.path.exists(newclip_resultseq_path):
                os.makedirs(newclip_resultseq_path)

            # Load New Clip Data
            newclip_filepath = os.path.join(newclip_path, 'Clip-{:d}_nframe-{:d}_gap-{:s}.h5'.
                                            format(clip_i, N, gap_str))


            Data = deepdish.io.load(newclip_filepath)['Data']

            Px = np.array(Data['x'])
            Py = np.array(Data['y'])
            y_cat = np.array(Data['label'])

            ## Skip if less than 20 points
            if Px.shape[0] < 20:
                continue

            X_te = np.concatenate([Px, Py], axis=1)

            y_onehot_te = np.zeros(shape=[y_cat.shape[0], 10])
            for i in range(y_cat.shape[0]):
                y_onehot_te[i, y_cat[i]] = 1


        loss_te, y_hat_te = sess.run(
            [loss, y_hat],
            feed_dict={H_ph: np.expand_dims(X_te, axis=0), y_gt_ph: np.expand_dims(y_onehot_te,axis=0),
                       Is_Training_ph: False})

        loss_te_all += loss_te

        # Nc = len(np.unique(y_cat))
        Nc_range = [1,2,3,4,5,6,7,8,9]
        pred_te, Nc = MdlSel.Kmeans_SOD(y_hat_te[0, ...], Nc_range)
        acc_te = Tool.EvalAcc_Fast(pred_te, y_cat)

        acc_te_all.append(acc_te)
        pred_te_all.append(pred_te)

        acc_clip.append(acc_te)

        print('\rtest seq-{:s} acc-{:.2f}'.format(seq_name,acc_te),end='')

        ### Write prediction into FBMS format
        pred_filepath = os.path.join(PRED_PATH, 'Pred_seq-{:s}.dat'.format(seq_name))
        ValidTraj = []
        for i in range(0,pred_te.shape[0]):
            ValidTraj.append({'frame':list(range(1,11)),'x':Data['x'][i],'y':Data['y'][i]})

        FIO.ExportPred(ValidTraj, pred_te, seq_len, pred_filepath)

        ## Visualize the MoSeg results
        # Visualize every frame
        bmf_filepath = os.path.join(gt_path, '{:s}.bmf'.format(seq_name))
        fid = open(bmf_filepath)
        temp = parse.parse('{:d} {:d}', fid.readline())
        frame_filenames = []
        for f_i in range(0, seq_len):
            line = fid.readline()
            try:
                frame_filenames.append(parse.parse('{}\n', line)[0])
            except:
                frame_filenames.append(line)

        fid.close()

        Px = np.array(Data['x'])
        Py = np.array(Data['y'])
        label = np.array(Data['label'])

        plt.figure(figsize=[6.4,9.6])

        for f_i in range(0, len(Data['frame'])):

            ## GT
            plt.subplot(2,1,1)
            # Visualize original frame
            frame_img = plt.imread(os.path.join(seq_path, '{:s}'.format(frame_filenames[Data['frame'][f_i] - 1])))
            plt.cla()
            plt.title('Ground-Truth')
            plt.imshow(frame_img)

            # Plot overlayed feature points
            for m_i in np.unique(label):
                plt.plot(Px[label == m_i, f_i], Py[label == m_i, f_i], '.')


            ## Predicted
            plt.subplot(2,1,2)
            # Visualize original frame
            frame_img = plt.imread(os.path.join(seq_path, '{:s}'.format(frame_filenames[Data['frame'][f_i] - 1])))
            plt.cla()
            plt.title('Prediction')
            plt.imshow(frame_img)

            # Plot overlayed feature points
            for m_i in np.unique(pred_te):
                plt.plot(Px[pred_te == m_i, f_i], Py[pred_te == m_i, f_i], '.')

            # Save visualization to new clip sequence folder
            newclip_seq_filepath = os.path.join(newclip_resultseq_path, 'moseg_MdlSelSOD_{:03d}.png'.format(f_i))
            plt.savefig(newclip_seq_filepath)

        plt.close()

    ## Calculate Per Sequence Error
    acc_te_seq.append(np.mean(acc_clip))

## Save Per sequence error
perseq_error_filepath = os.path.join(SUMMARY_PATH,'PerSeqErr_MdlSelSOD.mat')
scio.savemat(perseq_error_filepath,{'acc_te_seq':acc_te_seq,'acc_te_all':acc_te_all})


print('Average MissClassificationAcc - {:.2f}'.format(100*np.mean(acc_te_all)))