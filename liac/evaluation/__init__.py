import time
import liac

import sklearn
import sklearn.cross_validation
import sklearn.feature_extraction
import sklearn.metrics
import sklearn.pipeline
import sklearn.grid_search

# CROSS VALIDATION ============================================================
Bootstrap               = sklearn.cross_validation.Bootstrap
KFold                   = sklearn.cross_validation.KFold
LeaveOneLabelOut        = sklearn.cross_validation.LeaveOneLabelOut
LeaveOneOut             = sklearn.cross_validation.LeaveOneOut
LeavePLabelOut          = sklearn.cross_validation.LeavePLabelOut
LeavePOut               = sklearn.cross_validation.LeavePOut
StratifiedKFold         = sklearn.cross_validation.StratifiedKFold
ShuffleSplit            = sklearn.cross_validation.ShuffleSplit
StratifiedShuffleSplit  = sklearn.cross_validation.StratifiedShuffleSplit
train_test_split        = sklearn.cross_validation.train_test_split
cross_val_score         = sklearn.cross_validation.cross_val_score
permutation_test_score  = sklearn.cross_validation.permutation_test_score
check_cv                = sklearn.cross_validation.check_cv
# =============================================================================

# FEATURE SELECTION ===========================================================
SelectPercentile        = sklearn.feature_selection.SelectPercentile
SelectKBest             = sklearn.feature_selection.SelectKBest
SelectFpr               = sklearn.feature_selection.SelectFpr
SelectFdr               = sklearn.feature_selection.SelectFdr
SelectFwe               = sklearn.feature_selection.SelectFwe
RFE                     = sklearn.feature_selection.RFE
RFECV                   = sklearn.feature_selection.RFECV
chi2                    = sklearn.feature_selection.chi2
f_classif               = sklearn.feature_selection.f_classif
f_regression            = sklearn.feature_selection.f_regression
# =============================================================================

# METRICS =====================================================================
accuracy_score                     = sklearn.metrics.accuracy_score
auc                                = sklearn.metrics.auc
average_precision_score            = sklearn.metrics.average_precision_score
classification_report              = sklearn.metrics.classification_report
confusion_matrix                   = sklearn.metrics.confusion_matrix
f1_score                           = sklearn.metrics.f1_score
fbeta_score                        = sklearn.metrics.fbeta_score
# hamming_loss                       = sklearn.metrics.hamming_loss
hinge_loss                         = sklearn.metrics.hinge_loss
# jaccard_similarity_score           = sklearn.metrics.jaccard_similarity_score
# log_loss                           = sklearn.metrics.log_loss
matthews_corrcoef                  = sklearn.metrics.matthews_corrcoef
precision_recall_curve             = sklearn.metrics.precision_recall_curve
precision_recall_fscore_support    = sklearn.metrics.precision_recall_fscore_support
precision_score                    = sklearn.metrics.precision_score
recall_score                       = sklearn.metrics.recall_score
# roc_auc_score                      = sklearn.metrics.roc_auc_score
roc_curve                          = sklearn.metrics.roc_curve
zero_one_loss                      = sklearn.metrics.zero_one_loss
explained_variance_score           = sklearn.metrics.explained_variance_score
mean_absolute_error                = sklearn.metrics.mean_absolute_error
mean_squared_error                 = sklearn.metrics.mean_squared_error
r2_score                           = sklearn.metrics.r2_score
adjusted_mutual_info_score         = sklearn.metrics.adjusted_mutual_info_score
adjusted_rand_score                = sklearn.metrics.adjusted_rand_score
completeness_score                 = sklearn.metrics.completeness_score
homogeneity_completeness_v_measure = sklearn.metrics.homogeneity_completeness_v_measure
homogeneity_score                  = sklearn.metrics.homogeneity_score
mutual_info_score                  = sklearn.metrics.mutual_info_score
normalized_mutual_info_score       = sklearn.metrics.normalized_mutual_info_score
silhouette_score                   = sklearn.metrics.silhouette_score
silhouette_samples                 = sklearn.metrics.silhouette_samples
v_measure_score                    = sklearn.metrics.v_measure_score
# =============================================================================

# PIPELINE ====================================================================
Pipeline                = sklearn.pipeline.Pipeline
FeatureUnion            = sklearn.pipeline.FeatureUnion
# =============================================================================

# GRID SEARCH =================================================================
GridSearchCV            = sklearn.grid_search.GridSearchCV
# ParameterGrid           = sklearn.grid_search.ParameterGrid
# ParameterSampler        = sklearn.grid_search.ParameterSampler
# RandomizedSearchCV      = sklearn.grid_search.RandomizedSearchCV
# =============================================================================

class Timer(object):
    def __init__(self):
        self.t_start = time.time()
        self.t_best = liac.constants.inf
        self.t_worse = -liac.constants.inf

    def get_elapsed(self):
        return time.time() - self.t_start

    def tic(self):
        self.t_start = time.time()

    def toc(self):
        toc = self.get_elapsed()
        if toc > self.t_worse:
            self.t_worse = toc
        if toc < self.t_best:
            self.t_best = toc
        return toc

    def __repr__(self):
        return 'Timer elapsed: %.4f'%self.get_elapsed()