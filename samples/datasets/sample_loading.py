import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import liac

print liac.dataset.load('iris'), '\n\n'
print liac.dataset.load('linaker'), '\n\n'
print liac.dataset.load('sets/xor.arff'), '\n\n'
print liac.dataset.load('sets/test.csv'), '\n\n'

