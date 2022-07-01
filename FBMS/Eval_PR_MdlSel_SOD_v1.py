import os
import sys
import parse
import numpy as np
import matplotlib.pyplot as plt
import argparse as arg

sys.path.append('../Util')

import EvalPR

# Para = {}
# Para['depth'] = 50
# lossname = 'L2Loss'
# FixedLen = 10
# nCV = 22
# HiddenDim = 256


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
parser.add_argument('--RsltPath',type=str,help='Result path',default='/home/xuxun/Dropbox/GitHub/LearnSubspaceMoSeg/Results/FBMS/CorresNet50_L2Loss_fixedLen-10_HiddenDim-256_Results')


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

seq_te = 'cars1,cars4,cars5,cars10,marple2,marple4,marple6,marple7,marple9,marple12,people1,people2,tennis,camel01,cats01,cats03,cats06,dogs01,dogs02,farm01,goats01,horses02,horses04,horses05,lion01,people03,giraffes01,rabbits02,rabbits03,rabbits04'
# seq_te = 'horses04'
# seq_te = 'marple7,marple9,marple12,people1,people2,tennis,camel01,cats01,cats03,cats06,dogs01,dogs02,farm01,goats01,horses02,horses04,horses05,lion01,people03,giraffes01,rabbits02,rabbits03,rabbits04'
seqlist_te = seq_te.split(',')

density = []
Precision = []
Recall = []
Fmeasure = []
Precision_my = []
Recall_my = []
Fmeasure_my = []

for seq_name in seqlist_te:

    if seq_name == 'camel01':
        a=1

    ## Directories
    base_path = os.path.abspath(
        '../Dataset/FBMS/Data/{}'.format(seq_name))
    gt_path = os.path.abspath('../Dataset/FBMS/Data/{:s}'.format(seq_name))
    seq_path = os.path.abspath('../Dataset/FBMS/Data/{:s}/Seq'.format(seq_name))
    newclip_path = base_path


    evalbin_filepath = '/home/elexuxu/vision/Alex/Data/FBMS/moseg/evalcode/MoSegEvalPR'
    gt_filepath = os.path.join(gt_path, '{:s}Def.dat'.format(seq_name))
    pred_filepath = os.path.join(PRED_PATH,'Pred_seq-{:s}.dat'.format(seq_name))

    mTracks, mClusterNo = EvalPR.readTracks(pred_filepath)
    Mask, LabelMask, FrameIndex, _ = EvalPR.LoadGT(gt_path, seq_name)
    # EvalPR.Precision(mTracks,mClusterNo,Mask,Frame)

    # labelled pixels on all labelled frames
    LabelledPixel = {'x': [], 'y': [], 'pred': [], 'gt': []}

    for f_i in range(1,FixedLen+1):

    # for f_i in range(1, 2):
        # check all frames
        if f_i in FrameIndex:
            # labelled frame
            FrameLabelMask = LabelMask[FrameIndex.index(f_i)]

            ## Collect Labelled Pixels Predicted by Model

            for t_i in range(0, len(mTracks)):
                # accumulate each trajectory point
                Traj = mTracks[t_i]['mPoints']
                Pred = mTracks[t_i]['mLabel']
                for tf_i in range(0, len(Traj)):
                    # collect prediction & gt on labelled frame
                    if Traj[tf_i]['frame'] == f_i:
                        LabelledPixel['x'].append(Traj[tf_i]['x'])
                        LabelledPixel['y'].append(Traj[tf_i]['y'])
                        LabelledPixel['pred'].append(Pred)

                        # get the gt label
                        LabelledPixel['gt'].append(FrameLabelMask[Traj[tf_i]['y'], Traj[tf_i]['x']])

                        break

    ## Calculate PR and Fmeasure
    UniquePred = np.unique(LabelledPixel['pred'])
    nPredClust = len(UniquePred)

    UniqueGT = np.unique(LabelledPixel['gt'])
    nGTClust = len(UniqueGT)

    P = np.zeros([nPredClust, nGTClust])
    R = P.copy()
    Fm = P.copy()

    for i in range(0, nPredClust):
        # all predicted clusters
        ci = np.where(LabelledPixel['pred'] == UniquePred[i])[0]

        for j in range(0, nGTClust):
            # all gt clusters
            gj = np.where(LabelledPixel['gt'] == UniqueGT[j])[0]

            try:
                P[i, j] = np.intersect1d(ci, gj).shape[0] / (ci.shape[0] + 1e-6)
            except:
                a = 1
            try:
                R[i, j] = np.intersect1d(ci, gj).shape[0] / (gj.shape[0] + 1e-6)
            except:
                a = 1
            if R[i, j] == 0 and P[i, j] == 0:
                Fm[i, j] = 0
            else:
                Fm[i, j] = 2 * P[i, j] * R[i, j] / (P[i, j] + R[i, j])

    ## Fill empty clusters if there are less predicted clusters than gt
    if nPredClust < nGTClust:
        P = np.concatenate([P, np.ones(shape=[nGTClust - nPredClust, P.shape[1]])], axis=0)
        R = np.concatenate([R, np.zeros(shape=[nGTClust - nPredClust, R.shape[1]])], axis=0)
        Fm = np.concatenate([Fm, np.zeros(shape=[nGTClust - nPredClust, Fm.shape[1]])], axis=0)

    ## Find best assignment with Hungarian Method
    from scipy.optimize import linear_sum_assignment as HungarianAlg

    Assign = HungarianAlg((-10000 * Fm).astype(int))

    # get the P,R&Fm under optimal assignment
    avgP = 0.
    avgR = 0.
    cnt = 0.
    for i in range(0, len(Assign[0])):
        r = Assign[0][i]
        c = Assign[1][i]
        avgP += P[r, c]
        avgR += R[r, c]
        cnt += 1
    avgP /= cnt
    avgR /= cnt
    avgF = 2.0 * avgP * avgR / (avgP + avgR)

    Precision_my.append(avgP)
    Recall_my.append(avgR)
    Fmeasure_my.append(avgF)

    ##### Compare with Ochs' code

    # command = '{:s} {:s} {:s}'.format(evalbin_filepath,gt_filepath,pred_filepath)
    #
    # os.system(command)
    #
    # # Check Results
    # eval_filepath = os.path.join(PRED_PATH,'Pred_seq-{:s}Numbers.txt'.format(seq_name))
    #
    # fid = open(eval_filepath,'r')
    #
    # contents = fid.readlines()
    #
    # Failed = False
    #
    # for l_i in range(0,len(contents)):
    #     line = contents[l_i]
    #     if 'Average region' in line:
    #         try: density.append(parse.parse('{:f}\n',contents[l_i+1])[0])
    #         except:
    #             print('Failed {:s}'.format(seq_name))
    #             Failed=True
    #             break
    #
    #     if 'Average Precision, Recall, F-measure' in  line:
    #
    #         tmp = parse.parse('{:f} {:f} {:f}\n',contents[l_i+1])
    #         Precision.append(tmp[0])
    #         Recall.append(tmp[1])
    #         Fmeasure.append(tmp[2])
    #
    # fid.close()

    print('My Implementation seq-{:s} P-{:.2f} R-{:.2f} Fm-{:.2f}'.format(seq_name, avgP, avgR, avgF))
    # print('Ochs'' Implementation P-{:.2f} R-{:.2f} Fm-{:.2f}'.format(Precision[-1],Recall[-1],Fmeasure[-1]))

# print('Average Density {:.2f}'.format(np.mean(density)))
# print('Average Precision {:.2f}'.format(100*np.mean(Precision)))
# print('Average Recall {:.2f}'.format(100*np.mean(Recall)))
# print('Average Fmeasure {:.2f}'.format(100*np.mean(Fmeasure)))

FmeasureAvg_my = 2. * np.mean(Precision_my) * np.mean(Recall_my) / (np.mean(Precision_my) + np.mean(Recall_my))

print('My Implementation Average Precision {:.2f}'.format(100 * np.mean(Precision_my)))
print('My Implementation Average Recall {:.2f}'.format(100 * np.mean(Recall_my)))
print('My Implementation Average Fmeasure {:.2f}'.format(100 * FmeasureAvg_my))