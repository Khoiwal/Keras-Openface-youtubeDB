# This code is based on Facenet Impelmentation by David Sandberg, more specifically the code file validate_on_lfw.py. The repository is at below location
#https://github.com/davidsandberg/facenet/blob/master/src/validate_on_lfw.py
#Additionally the code also references Victor Sy Wang's implementation, specifically for OpenKeras 
# at https://github.com/iwantooxxoox/Keras-OpenFace/blob/master/Keras-Openface-Accuracy.ipynb
# MIT License
# 
# Copyright (c) 2016 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import tensorflow as tf
from keras.models import load_model
from keras.utils import CustomObjectScope
with CustomObjectScope({'tf': tf}):
     model = load_model('./model/nn4.small2.v1.h5')
import os
import numpy as np
import argparse
import math
import facenet
import youtubedb
import time
import sys
import math
import matplotlib.pyplot as plt
from sklearn import metrics
from scipy.optimize import brentq
from scipy import interpolate
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from itertools import cycle

def main(args):

    # Read the file containing the pairs used for testing
    pairs = youtubedb.read_pairs(os.path.expanduser(args.youtubedb_pairs))

    # Get the paths for the corresponding images
    paths, actual_issame = youtubedb.get_paths(os.path.expanduser(args.youtubedb_dir), pairs)
    batch_size = args.batch_size
    embedding_size=128
    image_size=args.image_size
    nrof_images = len(paths)
    print("No of images:")
    print(nrof_images)
    nrof_batches = int(math.ceil(1.0*nrof_images / batch_size))
    emb_array = np.zeros((nrof_images, embedding_size))

    for i in range(nrof_batches):
        start_index = i*batch_size
        end_index = min((i+1)*batch_size, nrof_images)
        paths_batch = paths[start_index:end_index]
        images = facenet.load_data(paths_batch, False, False, image_size)

        t0 = time.time()
        y = model.predict(images)
        emb_array[start_index:end_index,:] = y
        t1 = time.time()

        print('batch: ', i, ' time: ', t1-t0)
        print(emb_array)
    
    tpr, fpr, accuracy, val, val_std, far =  youtubedb.evaluate(emb_array, 
                actual_issame, nrof_folds=args.nrof_folds)

    print('Accuracy: %1.3f+-%1.3f' % (np.mean(accuracy), np.std(accuracy)))
    print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))
    auc = metrics.auc(fpr, tpr)
    print('Area Under Curve (AUC): %1.3f' % auc)
    eer = brentq(lambda x: 1. - x - interpolate.interp1d(fpr, tpr)(x), 0., 1.)
    print('Equal Error Rate (EER): %1.3f' % eer)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', linewidth=lw, label='ROC curve (area = %0.2f)' % auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()        
    
       
def parse_arguments(argv):
     parser = argparse.ArgumentParser()
    
     parser.add_argument('youtubedb_dir', type=str,
        help='Path to the data directory containing aligned Youtubedb face patches.')
     parser.add_argument('--batch_size', type=int,
        help='Number of images to process in a batch in the YoutubeDB test set.', default=100)
     parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=96)
     parser.add_argument('--youtubedb_pairs', type=str,
        help='The file containing the pairs to use for validation.', default='data/YoutubeDBpairs.txt')
     parser.add_argument('--nrof_folds', type=int,
        help='Number of folds to use for cross validation. Mainly used for testing.', default=10)
     return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

