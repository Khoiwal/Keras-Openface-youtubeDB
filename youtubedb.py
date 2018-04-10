# Copyright (c) 2018 Hrishi
#Helper for evaluation on the Youtube faces dataset 
#This code has been updated to cater for the different directory structure within the class pairs file used for evaluation



# MIT License

# 

# Copyright (c) 2018 Hrishi
# This code reuses the code from David Sandberg for performance evaluation on labelled faces in the wild.
# https://github.com/davidsandberg/facenet/blob/master/src/lfw.py
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



from __future__ import absolute_import

from __future__ import division

from __future__ import print_function



import os

import numpy as np

import facenet



def evaluate(embeddings, actual_issame, nrof_folds=10):

    # Calculate evaluation metrics

    thresholds = np.arange(0, 4, 0.01)

    embeddings1 = embeddings[0::2]

    embeddings2 = embeddings[1::2]

    tpr, fpr, accuracy = facenet.calculate_roc(thresholds, embeddings1, embeddings2,

        np.asarray(actual_issame), nrof_folds=nrof_folds)

    thresholds = np.arange(0, 4, 0.001)

    val, val_std, far = facenet.calculate_val(thresholds, embeddings1, embeddings2,

        np.asarray(actual_issame), 1e-3, nrof_folds=nrof_folds)

    return tpr, fpr, accuracy, val, val_std, far



def get_paths(youtubedb_dir, pairs):

    nrof_skipped_pairs = 0

    path_list = []

    issame_list = []

    for pair in pairs:

        if os.path.exists(os.path.join(os.path.abspath(youtubedb_dir),pair[2].strip())):
            filepath0 = os.listdir(os.path.join(os.path.abspath(youtubedb_dir),pair[2].strip()))[0]
            path0 = os.path.join(os.path.abspath(youtubedb_dir), pair[2].strip(), filepath0)
            
           
        if os.path.exists(os.path.join(os.path.abspath(youtubedb_dir),pair[3].lstrip())):
            filepath1 = os.listdir(os.path.join(os.path.abspath(youtubedb_dir),pair[3].strip()))[0] 
            path1 = os.path.join(os.path.abspath(youtubedb_dir), pair[3].strip(), filepath1)

        print("Path Zero")
        print(path0)
        print("Path One")
        print(path1)    
        if  int(pair[4].strip())==1:
           
            issame = True

        else:    

            issame = False

        if os.path.exists(path0) and os.path.exists(path1):    # Only add the pair if both paths exist

            path_list += (path0,path1)

            issame_list.append(issame)

            
            print("Path Zero")
            print(path0)
            print("Path One")
            print(path1)
            print("Issame")
            print(issame)    

        else:

            nrof_skipped_pairs += 1

    if nrof_skipped_pairs>0:

        print('Skipped %d image pairs' % nrof_skipped_pairs)

    

    return path_list, issame_list



def read_pairs(pairs_filename):

    pairs = []

    with open(pairs_filename, 'r') as f:

        for line in f.readlines()[1:]:

            pair = line.strip().split(',')

            pairs.append(pair)

    return np.array(pairs)

