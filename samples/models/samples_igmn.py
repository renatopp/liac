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


'''
Sample: Learning Sine Function Using IGMN
=========================================

In this sample we use an IGMN to learn a Sine function. This sample aims to 
show graphically and step-be-step how an IGMN learn.

'''

import numpy as np
import liac
from liac import plot

# DATA SETUP ==================================================================
X = np.linspace(0, 2*np.pi, 100) # 100 points from 0 to 2*pi
Y = np.sin(X)
# =============================================================================

# IGMN SETUP ==================================================================
distance = np.array([2*np.pi, 2])
igmn = liac.models.IGMN(distance, delta=0.1, tau=0.1)
# =============================================================================

# LEARNING THE FUNCTION =======================================================
for x, y in zip(X, Y):
    # IGMN must always receive a numpy array
    igmn.learn(np.array([x, y]))

    # Plot step-by-step the IGMN to show the learning process
    liac.plot.clf()
    liac.plot.plot(X, Y, 'b') # plot all data
    igmn.plot(6) # plot IGMN clusters
    liac.plot.plot(x, y, 'ro') # plot the lest points presented to IGMN
    liac.plot.axis([-2.5, 2*np.pi+2.5, -3.5, 3.5]) # correct the axis
    liac.plot.pause(0.1)
# =============================================================================

# last plot.show to prevent the plot to close at the end of the training
liac.plot.show() 