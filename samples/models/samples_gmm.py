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

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import liac
from liac import plot


###############################################################################
# Generate sample data
centers = [[1, 1], [0.5, 0.5], [1, -1]]
X, idx = liac.random.make_gaussian(n_samples=100, centers=centers)

gmm = liac.models.GMM(3, 'full')
gmm.fit(X)

means = gmm.means_
covs = gmm.covars_
n_clusters_ = len(means)
ax = liac.plot.gca()

for i in xrange(n_clusters_):
    pi = idx == i
    plot.scatter(X[pi,0], X[pi,1], color=liac.random.make_color(i+10))

    e = liac.plot.Gaussian(means[i], covs[i], 5, color=liac.random.make_color(i), alpha=0.75)
    ax.add_artist(e)
    x, y = means[i]
    liac.plot.plot(x, y, 'x', markersize=14, markeredgewidth=2, color='k')
    liac.plot.plot(x, y, 'x', markersize=12, markeredgewidth=2, color=liac.random.make_color(i))

# for i, center in enumerate(centers):
    # X, idx = liac.random.make_gaussian(n_samples=100, centers=center)
#     plot.scatter(X[:,0], X[:,1], color=liac.random.make_color(i))
plot.show()

