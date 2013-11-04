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

__all__ = ['KMeans', 'PCA', 'PPCA', 'KernelPCA', 'GMM', 'KNN']


import sklearn
import sklearn.cluster
import sklearn.decomposition
import sklearn.mixture
import sklearn.neighbors
import sklearn.ensemble
import sklearn.hmm
import sklearn.naive_bayes
import sklearn.svm
import sklearn.gaussian_process
from . import igmn

# CLUSTER =====================================================================
KMeans                      = sklearn.cluster.KMeans

AffinityPropagation         = sklearn.cluster.AffinityPropagation
DBSCAN                      = sklearn.cluster.DBSCAN
MiniBatchKMeans             = sklearn.cluster.MiniBatchKMeans
MeanShift                   = sklearn.cluster.MeanShift
SpectralClustering          = sklearn.cluster.SpectralClustering
Ward                        = sklearn.cluster.Ward
# =============================================================================

# MATRIX DECOMPOSITION ========================================================
PCA                         = sklearn.decomposition.PCA
PPCA                        = sklearn.decomposition.ProbabilisticPCA
KernelPCA                   = sklearn.decomposition.KernelPCA

ProjectedGradientNMF        = sklearn.decomposition.ProjectedGradientNMF
RandomizedPCA               = sklearn.decomposition.RandomizedPCA
FactorAnalysis              = sklearn.decomposition.FactorAnalysis
FastICA                     = sklearn.decomposition.FastICA
# TruncatedSVD                = sklearn.decomposition.TruncatedSVD
NMF                         = sklearn.decomposition.NMF
SparsePCA                   = sklearn.decomposition.SparsePCA
MiniBatchSparsePCA          = sklearn.decomposition.MiniBatchSparsePCA
SparseCoder                 = sklearn.decomposition.SparseCoder
DictionaryLearning          = sklearn.decomposition.DictionaryLearning
MiniBatchDictionaryLearning = sklearn.decomposition.MiniBatchDictionaryLearning
# =============================================================================

# ENSEMBLE METHODS ============================================================
RandomForestClassifier      = sklearn.ensemble.RandomForestClassifier
RandomTreesEmbedding        = sklearn.ensemble.RandomTreesEmbedding
RandomForestRegressor       = sklearn.ensemble.RandomForestRegressor
ExtraTreesClassifier        = sklearn.ensemble.ExtraTreesClassifier
ExtraTreesRegressor         = sklearn.ensemble.ExtraTreesRegressor
# AdaBoostClassifier          = sklearn.ensemble.AdaBoostClassifier
# AdaBoostRegressor           = sklearn.ensemble.AdaBoostRegressor
GradientBoostingClassifier  = sklearn.ensemble.GradientBoostingClassifier
GradientBoostingRegressor   = sklearn.ensemble.GradientBoostingRegressor
# =============================================================================

# HIDDEN MARKOV MODELS ========================================================
GaussianHMM                 = sklearn.hmm.GaussianHMM
MultinomialHMM              = sklearn.hmm.MultinomialHMM
GMMHMM                      = sklearn.hmm.GMMHMM
# =============================================================================

# MIXTURE MODELS ==============================================================
IGMN                        = igmn.IGMN
GMM                         = sklearn.mixture.GMM
DPGMM                       = sklearn.mixture.DPGMM
VBGMM                       = sklearn.mixture.VBGMM
# =============================================================================

# NAIVE BAYES =================================================================
GaussianNB                  = sklearn.naive_bayes.GaussianNB
MultinomialNB               = sklearn.naive_bayes.MultinomialNB
BernoulliNB                 = sklearn.naive_bayes.BernoulliNB
# =============================================================================

# =============================================================================
KNN                         = sklearn.neighbors.NearestNeighbors
KNNClassifier               = sklearn.neighbors.KNeighborsClassifier
RadiusNeighborsClassifier   = sklearn.neighbors.RadiusNeighborsClassifier
KNeighborsRegressor         = sklearn.neighbors.KNeighborsRegressor
RadiusNeighborsRegressor    = sklearn.neighbors.RadiusNeighborsRegressor
NearestCentroid             = sklearn.neighbors.NearestCentroid
BallTree                    = sklearn.neighbors.BallTree
# KDTree                      = sklearn.neighbors.KDTree
# DistanceMetric              = sklearn.neighbors.DistanceMetric
KernelDensity               = sklearn.neighbors.KernelDensity
KDE                         = sklearn.neighbors.KernelDensity
# =============================================================================

# =============================================================================
SVC                         = sklearn.svm.SVC
LinearSVC                   = sklearn.svm.LinearSVC
NuSVC                       = sklearn.svm.NuSVC
SVR                         = sklearn.svm.SVR
NuSVR                       = sklearn.svm.NuSVR
OneClassSVM                 = sklearn.svm.OneClassSVM
l1_min_c                    = sklearn.svm.l1_min_c
# =============================================================================

# =============================================================================
GaussianProcess             = sklearn.gaussian_process.GaussianProcess
correlation_models          = sklearn.gaussian_process.correlation_models
regression_models           = sklearn.gaussian_process.regression_models
# =============================================================================
