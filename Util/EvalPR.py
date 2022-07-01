import numpy as np
import parse
import os
import matplotlib.pyplot as plt

def readTracks(aFilename):

    fid = open(aFilename,'r')

    aLength = parse.parse('{:d}',fid.readline())[0]
    aTrackNo = parse.parse('{:d}',fid.readline())[0]

    mTrackLabel = []
    aSize = 0
    mTracks = []

    for i in range(0,aTrackNo):
        tmp = parse.parse('{:d} {:d}',fid.readline())
        Label = tmp[0]
        aSize = tmp[1]

        mPoints = []
        for j in range(0,aSize):
            tmp = parse.parse('{} {} {:d}', fid.readline())
            x = tmp[0]
            y = tmp[1]
            frame = tmp[2]
            mPoints.append({'x':int(float(x)+0.5),'y':int(float(y)+0.5),'frame':frame})

        mTracks.append({'mPoints':mPoints,'mLabel':Label})

    mClusterNo = len(np.unique([mTracks[i]['mLabel'] for i in range(0,len(mTracks))]))

    return mTracks, mClusterNo


def LoadGT(gt_path,seq_name):
    ###### Load GT
    ## Load dat file
    Mask = []
    Frame = []
    Label = []
    UniqueVals = []

    dat_filepath = os.path.join(gt_path, '{:s}Def.dat'.format(seq_name))

    fid = open(dat_filepath, 'r')

    content = fid.readlines()

    for l_i in range(0, len(content)):

        line = content[l_i]

        if 'Total number of frames in this shot' in line:
            seq_len = parse.parse('{:d}\n', content[l_i + 1])[0]

        if 'Total number of labeled frames' in line:
            num_label_frames = parse.parse('{:d}\n', content[l_i + 1])[0]

        if 'Scale of region' in line:
            UniqueVals.append(parse.parse('{:d}\n', content[l_i+1])[0])

        if 'File name' in line:
            ## load the mask file/image
            gt_filepath = os.path.join(gt_path,
                                       parse.parse('{}\n', content[l_i + 1])[0])
            frame_no = parse.parse('{:d}\n', content[l_i - 1])[0] + 1
            img = plt.imread(gt_filepath)
            img = img.astype(int)

            format = gt_filepath.split('.')[-1]

            if format == 'ppm':
                MaskFrame = img[...,0]*65536 + img[...,1]*256 + img[...,2]
            else:
                MaskFrame = img

            # convert mask values to labels
            LabelFrame = np.zeros_like(MaskFrame,dtype=int)
            # UniqueVals = np.unique(MaskFrame)
            for newlabel in range(0,len(UniqueVals)):
                val = UniqueVals[newlabel]
                LabelFrame += newlabel * (MaskFrame==val).astype(int)


            Mask.append(MaskFrame)
            Label.append(LabelFrame)
            Frame.append(frame_no)

    fid.close()

    return Mask, Label, Frame, seq_len


def Precision(mTracks,mClusterNo,Mask,Frame):

    a=1