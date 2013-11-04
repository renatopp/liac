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

'''
This module is an interface to pandas and provides some utility functions for 
handling dataset.
'''

import os
import pandas as pd
from . import arff

__all__ = ['load', 'read_csv', 'read_clipboard', 'read_arff']

read_csv = pd.read_csv
read_clipboard = pd.read_clipboard

def read_arff(set_name):
    '''
    Read ARFF file into pandas DataFrame.

    :param set_name: the dataset path.
    '''
    f = open(set_name)
    info = arff.load(f)
    f.close()

    attributes = [a[0] for a in info['attributes']]
    data = info['data']
    return pd.DataFrame(data, columns=attributes)

def load(set_name, *args, **kwargs):
    '''
    This function loads automatically any dataset in the following formats: 
    arff; csv; excel; hdf; sql; json; html; stata; clipboard; pickle. Moreover,
    it loads the default datasets such "iris" if the extension in `set_name` is
    unknown.

    :param set_name: the dataset path or the default dataset name.
    :returns: a `pd.DataFrame` object.
    '''
    _, ext = os.path.splitext(set_name)

    if ext == '.arff':
        loader = read_arff
    elif ext in ['.csv', '.txt']:
        loader = read_csv
    else:
        loader = __load_default_set

    dataset = loader(set_name, *args, **kwargs)
    return dataset

def __load_default_set(set_name):
    ALIASES = {'linaker':'linaker1v'}
    name = ''.join([ALIASES.get(set_name, set_name), '.arff'])
    file_name = os.path.join(os.path.dirname(__file__), 'sets', name)
    return read_arff(file_name)
