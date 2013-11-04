# =============================================================================
# Federal University of Rio Grande do Sul (UFRGS)
# Connectionist Artificial Intelligence Laboratory (LIAC)
# Renato de Pontes Pereira - rppereira@inf.ufrgs.br
# =============================================================================
# Copyright (c) 2011 Renato de Pontes Pereira, renato.ppontes at gmail dot com
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy 
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights 
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell 
# copies of the Software, and to permit persons to whom the Software is 
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in 
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# =============================================================================

import numpy as np
import random
import sklearn
import sklearn.datasets

make_image                     = sklearn.datasets.load_sample_images
make_classification            = sklearn.datasets.make_classification
make_multilabel_classification = sklearn.datasets.make_multilabel_classification
make_regression                = sklearn.datasets.make_regression
make_blobs                     = sklearn.datasets.make_blobs
make_circles                   = sklearn.datasets.make_circles
make_friedman1                 = sklearn.datasets.make_friedman1
make_friedman2                 = sklearn.datasets.make_friedman2
make_friedman3                 = sklearn.datasets.make_friedman3
make_hastie_10_2               = sklearn.datasets.make_hastie_10_2
make_low_rank_matrix           = sklearn.datasets.make_low_rank_matrix
make_sparse_coded_signal       = sklearn.datasets.make_sparse_coded_signal
make_sparse_uncorrelated       = sklearn.datasets.make_sparse_uncorrelated
make_spd_matrix                = sklearn.datasets.make_spd_matrix
make_swiss_roll                = sklearn.datasets.make_swiss_roll
make_s_curve                   = sklearn.datasets.make_s_curve
make_sparse_spd_matrix         = sklearn.datasets.make_sparse_spd_matrix
# make_biclusters                = sklearn.datasets.make_biclusters
# make_checkerboard              = sklearn.datasets.make_checkerboard

def make_gaussian(n_samples=100, centers=(0, 0), scale=1):
    centers = np.atleast_2d(centers)
    n, d = np.shape(centers)
    data = []
    idx = []
    for i in xrange(n):
        mean = centers[i]
        S = np.random.randn(d, d)
        S = np.dot(S.T, S)
        C = np.linalg.cholesky(S).T

        gen = np.random.randn(n_samples, d).dot(C)
        max_ = np.max(gen, 0)
        min_ = np.min(gen, 0)
        gen = ((gen-min_)/(max_-min_) - 0.5)*2 + mean
        data.append(gen*scale)
        idx.append(np.ones(n_samples)*i)

    return np.concatenate(data), np.concatenate(idx)

def make_color(seed=None, range=1):
    random.seed(seed)
    r = (  random.random())*range
    g = (1-random.random())*range
    b = (  random.random())*range
    random.seed()
    return [r, g, b]