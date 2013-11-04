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
import scipy
import scipy.linalg
import sklearn

__all__ = ['IGMN']

EPS = np.finfo(float).eps
TINY = np.finfo(float).tiny

def mvnpdf(x, mu=None, sigma=None):
    '''Multivariate normal probability density function (pdf).'''

    dimension = len(x)

    if mu is None:
        mu = np.zeros(dimension)

    if sigma is None:
        sigma = np.eye(dimension)

    distance = x-mu
    determinant = np.linalg.det(sigma)
    inverse = np.linalg.inv(sigma)

    return np.exp(-0.5*(distance).dot(inverse).dot(distance)) * \
           1.0/((2*np.pi)**(dimension/2.0) * np.sqrt(determinant))


class IGMN(sklearn.base.BaseEstimator):
    '''
    The class for IGMN (Incremental Gaussian Mixture Network).

    .. NOTE: By convention the word "component" is used instead of "neuron".
    '''

    def __init__(self, distance=None, delta=0.1, tau=0.1, sp_min=None, 
                       v_min=None, tau_max=None, uniform=False):
        '''
        Creates a new IGMN instance. 

        The `distance` is the only mandatory parameter in this method. This 
        parameter represents the data range, i.e., ``max(data)-min(data)``;. 
        You can get the data range automatically using `liac.data_range`.

        The `delta` parameter is a fraction of the `distance` which will be 
        used to create the initial covariance matrices. In a practical view, 
        this parameter defines the size of the distributions.

        The `tau` and `tau_max` are thresholds which inform IGMN when to create
        and update the components, respectively. When a new input pattern is 
        presented to this model, the likelihood relative to the input and all
        components is computed. A given component can "absorbs" the input if it
        represents well enough the pattern, i.e., if its likelihood is greater 
        than `tau`. The `tau_max` is used to avoid over-fitting, restraining 
        the update for components when the likelihood is too big (e.g., >0.95).

        The `sp_min` and `v_min` parameters are used to remove noisy 
        components. When a new input pattern is presented to the model, it 
        verifies if some component is older than `v_min` and have less 
        activation than `sp_min`, if it is the case, the component is removed.

        The `uniform` parameter defines is the components are equiprobable or 
        not. I.e., if ``uniform = True``, all components will have the same 
        prior probability.

        :param distance: an 1xD numpy array.
        :param delta: an float between 0 and 1. Default to 0.1.
        :param tau: an float between 0 and 1. Default to 0.1.
        :param sp_min: a real number. Default to D+1.
        :param v_min: an integer. Default to D*2.
        :param tau_max: an float between 0 and 1, must be bigger than `tau`. 
                        Default to None.
        :param uniform: a boolean. Default to False.
        '''
        self.distance = distance
        self.dimension = distance.size
        self.size = 0
        self.n = 0
        
        self.priors = []
        self.means = []
        self.covs = []
        self.sps = []
        self.vs = []
        self.posts = []
        self.log_likes = []
        self.log_posts = []
        
        # CACHE ===============================================================
        self.cache_inverses = []
        self.cache_dets = []
        self.cache_distances = []
        self.cache_like = []
        # =====================================================================

        # PARAMS ==============================================================
        self.tau_max = tau_max
        self.delta = delta
        self.tau = tau
        self.sp_min = sp_min if sp_min is not None else self.dimension+1
        self.v_min = v_min if v_min is not None else 2*self.dimension
        self.initial_cov = np.diagflat((self.delta*self.distance)**2)
        self.min_cov = np.eye(self.dimension)*EPS
        self.uniform = uniform
        # =====================================================================

        # EXPERIMENTAL ========================================================
        self.able_to_updates = []
        # =====================================================================

    # INTERNAL METHODS ========================================================
    def log_mvnpdf(self, X, i):
        mu = self.means[i]
        cov = self.covs[i]

        n_dim = X.size
        cv_chol = scipy.linalg.cholesky(cov, lower=True)
        cv_log_det = 2 * np.sum(np.log(np.diagonal(cv_chol)))
        cv_sol = scipy.linalg.solve_triangular(cv_chol, (X - mu).T, lower=True).T
        log_prob = -.5*(np.sum(cv_sol**2) + n_dim*np.log(2*np.pi) + cv_log_det)
        self.cache_like[i] = -.5*np.sum(cv_sol**2)

        return log_prob

    def __mvnpdf(self, x, mean, cov):
        pdf = mvnpdf(x, mean, cov)

        if np.isnan(pdf): 
            pdf = 0
        
        return pdf + TINY

    def __pdf(self, x, i):
        dimension = self.dimension
        mu = self.means[i]
        cov = self.covs[i]

        if self.cache_inverses[i] is None:
            self.cache_inverses[i] = np.linalg.inv(cov)

        if self.cache_dets[i] is None:
            self.cache_dets[i] = np.linalg.det(cov)
            
        distance = x-mu
        inverse = self.cache_inverses[i]
        determinant = self.cache_dets[i]

        density = ((2*np.pi)**(dimension/2.0) * np.sqrt(determinant))
        self.cache_distances[i] = np.exp(-0.5*(distance).dot(inverse).dot(distance))
        pdf = self.cache_distances[i]/density

        if np.isnan(pdf): 
            pdf = 0
        
        return pdf + TINY

    def __get_index_by_features(self, features_a, features_b=None):
        if features_b is None: features_b = features_a
        return [[i*self.dimension+j for j in features_b] for i in features_a]

    def _compute_likelihood(self, x):
        self.log_likes = []
        self.cache_like = np.zeros(self.size)
        for i in xrange(self.size):
            self.log_likes.append(self.log_mvnpdf(x, i))

    def _compute_posterior(self):
        log_density = np.exp(self.log_likes + np.log(self.priors))
        self.posts = log_density/np.sum(log_density)


    def _has_acceptable_distribution(self):
        if not self.size:
            return False

        r = False
        min_like = np.log(self.tau)
        
        if self.tau_max is not None:
            max_like = np.log(self.tau_max)

        for i in xrange(self.size):
            if self.cache_like[i] >= min_like:
                r = True

                if self.tau_max is not None:
                    if self.cache_like[i] > max_like:
                        self.able_to_updates[i] = False
                    else:
                        self.able_to_updates[i] = True
                else:
                    self.able_to_updates[i] = True
        return r

    def _incremental_estimation(self, x):
        for i in xrange(self.size):
            self.vs[i] += 1
            self.sps[i] += self.posts[i]

            if not self.able_to_updates[i]: continue

            w = self.posts[i]/self.sps[i]
            wi = 1-w

            old_mean = self.means[i]
            old_cov = self.covs[i]

            delta = w*(x - old_mean);
            new_mean = old_mean + delta
            
            diff = x - new_mean
            new_cov = wi*old_cov - np.outer(delta, delta) + w*np.outer(diff, diff) + self.min_cov

            self.means[i] = new_mean
            self.covs[i] = new_cov
            self.cache_inverses[i] = None
            self.cache_dets[i] = None
            self.cache_distances[i] = None

    def _update_priors(self):
        if self.uniform:
            self.priors = [1./self.size]*self.size
        else:
            sp_sum = float(np.sum(self.sps))
            self.priors = [self.sps[i]/sp_sum for i in xrange(self.size)]

    def _delete_spurious(self):
        for i in reversed(xrange(self.size)):
            if self.vs[i] > self.v_min and self.sps[i] < self.sp_min:
                self.size -= 1
                del self.vs[i]
                del self.sps[i]
                del self.priors[i]
                del self.means[i]
                del self.covs[i]
                del self.cache_inverses[i]
                del self.cache_dets[i]
                del self.cache_distances[i]
                del self.able_to_updates[i]

    def _new_component(self, x, cov=None):
        if cov is None:
            cov = self.initial_cov.copy()
            
        self.size += 1
        self.able_to_updates.append(True)
        self.priors.append(1)
        self.sps.append(1)
        self.vs.append(1)
        self.means.append(x)
        self.covs.append(cov)
        self.cache_inverses.append(None)
        self.cache_dets.append(None)
        self.cache_distances.append(None)
    # =========================================================================

    def reset(self):
        '''
        Reset the model, removing all components but keeping the parameters.
        '''
        self.n = 0
        self.able_to_updates = []

        self.size = 0
        self.priors = []
        self.means = []
        self.covs = []
        self.sps = []
        self.vs = []
        self.log_likes = []
        self.posts = []
        self.cache_inverses = []
        self.cache_dets = []
        self.cache_densitys = []
        self.cache_distances = []

    def learn(self, x):
        '''
        Learns a single instance.

        :param x: an 1xD numpy array.
        :returns: a list with the posterior probabilities.
        '''
        self._compute_likelihood(x)
        if not self._has_acceptable_distribution():
            self._new_component(x)
            self._compute_likelihood(x)
            self._update_priors()
            
        self._compute_posterior()
        self._incremental_estimation(x)
        self._update_priors()
        self._delete_spurious()

        self.n += 1
        return self.posts

    def train(self, x, y=None):
        '''
        Learns a list of instances.

        If `y` is None, IGMN assume that the target values are concatenated to 
        the `x` vectors.

        :param x: an NxD numpy array with the prediction vectors.
        :param y: an NxD numpy array with the target vectors. 
        '''
        if y is None:
            for row in x:
                self.learn(row)
        else:
            for row_x, row_y in zip(x, y):
                self.learn(liac.concat(row_x, row_y))

    def fit(self, X):
        X = np.atleast_2d(X)
        for x in X:
            self.learn(x)

    def call(self, x):
        '''
        Computes the likelihood and posterior probabilities for a given input.

        :param x: an 1xD numpy array.
        :returns: a list with the posterior probabilities.
        '''
        self._compute_likelihood(x)
        self._compute_posterior()
        return self.posts

    def recall(self, x, features=None):
        '''
        Performs the Gaussian Mixture Regression for a given input.

        This method returns an estimate for the missing values. The `x` 
        parameter defines the known values while the `features` defines which
        attributes are know. Notice, if ``features is None``, IGMN assumes that
        the firsts M attributes are known, where M is the dimension of `x`.

        :param x: an 1xM numpy array.
        :param features: a list of integers. Default to None.
        :returns: a numpy array with the estimate for the missing values.
        '''
        features_a = list(features or range(x.size))
        features_b = list(set(range(self.dimension)) - set(features_a))

        inv = np.linalg.inv
        pjas = []
        xs = []

        for index in xrange(self.size):
            mean_a = self.means[index].take(features_a)
            mean_b = self.means[index].take(features_b)
            cov_a = self.covs[index].take(self.__get_index_by_features(features_a))
            cov_ab = self.covs[index].take(self.__get_index_by_features(features_a, features_b))
            
            pja_ = self.__mvnpdf(x, mean_a, cov_a)*self.priors[index]
            x_ = mean_b + np.dot(cov_ab.T, np.dot(inv(cov_a), x-mean_a))

            pjas.append(pja_)
            xs.append(x_)

        pjas = pjas/np.sum(pjas)
        return np.dot(pjas, xs)

    def classify(self, x):
        '''
        Classify a given input.

        It returns a label-binarized vector where the predict class have value 
        1.

        :param x: an 1xM numpy array.
        :returns: an array label-binarized.
        '''
        y_ = self.recall(x).tolist()
        i = y_.index(max(y_))
        y = np.zeros(self.dimension - x.size)
        y[i] = 1

        return y

    def get_best_component(self, x=None):
        '''
        Return the index of component with the largest likelihood.

        :return: an integer.
        '''
        if x is not None:
            self.call(x)
            
        return np.argmax(self.posts)

    def report(self):
        s = ''
        s += '-------------\n'
        s += 'Model Details\n'
        s += '-------------\n'
        s += '\n'
        s += 'Incremental Gaussian Mixture Network (IGMN)\n'
        s += '\n'
        s += 'Instances: ' + str(self.n)
        s += '\n'
        s += 'COMPONENT (%d)\n'%self.size
        # s += '\n'
        for i in xrange(self.size):
            s +=  '%4d - p(j): %.4f\n'%(i, self.priors[i])
            s += '     - mean: ' + liac.pr(self.means[i]) + '\n'
            s += '\n'
        s += '\n'
        return s

    def plot(self, *args, **kwargs):
        import liac
        if self.dimension == 1:
            for i in xrange(self.size):
                mean = self.means[i]
                var = self.covs[i]

                sigma = np.sqrt(var)
                x_min = args[0] if len(args) > 0 else mean-self.distance[0]
                x_max = args[1] if len(args) > 1 else mean+self.distance[0]

                X = np.linspace(x_min, x_max, 500)
                Y = liac.normpdf(X, mean, sigma)[0]

                Y = Y/np.max(Y)

                liac.plot.plot(X, Y, color=liac.random.make_color(i))

        elif self.dimension == 2:
            nstd = args[0] if len(args) > 0 else 2
            ax = liac.plot.gca()
            for i in xrange(self.size):
                ellipse = liac.plot.Gaussian(self.means[i], self.covs[i], nstd, color=liac.random.make_color(i), alpha=0.75)
                ax.add_artist(ellipse)
                x, y= self.means[i]
                liac.plot.plot(x, y, 'x', markersize=14, markeredgewidth=2, color='k')
                liac.plot.plot(x, y, 'x', markersize=12, markeredgewidth=2, color=liac.random.make_color(i))
                # liac.plot.add_ellipse(self.means[i], self.covs[i])

    def __repr__(self):
        return '<IGMN:%d:%d>'%(self.size, self.n)
    __str__ = __repr__
    __call__ = call
