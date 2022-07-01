import os
import sys
import parse
import numpy as np

sys.path.append('../Util')

import EvalPR

Para = {}
Para['depth'] = 50
lossname = 'L2Loss'
FixedLen = 10
nCV = 22
HiddenDim = 256

MODEL_NAME = 'CorresNet{:d}_{}'.format(Para['depth'],lossname)
DATA_DIR = os.path.expanduser('../Results/FBMS')
RESULT_DIR = '/home/xuxun/Dropbox/GitHub/LearnSubspaceMoSeg/Results/FBMS/CorresNet50_L2Loss_fixedLen-10_HiddenDim-256_Results'
CHECKPOINT_PATH = os.path.join(RESULT_DIR, 'CheckPoint')
SUMMARY_PATH = os.path.join(RESULT_DIR, 'Summary')
summary_filepath = os.path.join(SUMMARY_PATH, 'Summary.txt')
EMBED_PATH = os.path.join(RESULT_DIR,'Embedding')
PRED_PATH = os.path.join(RESULT_DIR,'Prediction')
MOSEG_PATH = os.path.join(RESULT_DIR,'MoSeg')

seq_te = 'cars1,cars4,cars5,cars10,marple2,marple4,marple6,marple7,marple9,marple12,people1,people2,tennis,camel01,cats01,cats03,cats06,dogs01,dogs02,farm01,goats01,horses02,horses04,horses05,lion01,people03,giraffes01,rabbits02,rabbits03,rabbits04'
# seq_te = 'marple7'
# seq_te = 'marple7,marple9,marple12,people1,people2,tennis,camel01,cats01,cats03,cats06,dogs01,dogs02,farm01,goats01,horses02,horses04,horses05,lion01,people03,giraffes01,rabbits02,rabbits03,rabbits04'
seqlist_te = seq_te.split(',')

density = []
Precision = []
Recall = []
Fmeasure = []

for seq_name in seqlist_te:

    if seq_name == 'camel01':
        a=1

    ## Directories
    base_path = os.path.abspath(
        '../Dataset/FBMS/Data/{}'.format(seq_name))
    gt_path = os.path.abspath('../Dataset/FBMS/Data/{:s}'.format(seq_name))
    seq_path = os.path.abspath('../Dataset/FBMS/Data/{:s}/Seq'.format(seq_name))
    newclip_path = base_path

    evalbin_filepath = '/vision02/Data/FBMS/moseg/evalcode/MoSegEvalPR'
    gt_filepath = os.path.join(gt_path, '{:s}Def.dat'.format(seq_name))
    pred_filepath = os.path.join(PRED_PATH,'Pred_seq-{:s}.dat'.format(seq_name))

    command = '{:s} {:s} {:s}'.format(evalbin_filepath,gt_filepath,pred_filepath)

    os.system(command)

    # Check Results
    eval_filepath = os.path.join(PRED_PATH,'Pred_seq-{:s}Numbers.txt'.format(seq_name))

    fid = open(eval_filepath,'r')

    contents = fid.readlines()

    Failed = False

    for l_i in range(0,len(contents)):
        line = contents[l_i]
        if 'Average region' in line:
            try: density.append(parse.parse('{:f}\n',contents[l_i+1])[0])
            except:
                print('Failed {:s}'.format(seq_name))
                Failed=True
                break

        if 'Average Precision, Recall, F-measure' in  line:

            tmp = parse.parse('{:f} {:f} {:f}\n',contents[l_i+1])
            Precision.append(tmp[0])
            Recall.append(tmp[1])
            Fmeasure.append(tmp[2])

    fid.close()


print('Average Density {:.2f}'.format(np.mean(density)))
print('Average Precision {:.2f}'.format(100*np.mean(Precision)))
print('Average Recall {:.2f}'.format(100*np.mean(Recall)))
print('Average Fmeasure {:.2f}'.format(100*np.mean(Fmeasure)))