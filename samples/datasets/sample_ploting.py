import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import liac
from liac import plot

data = liac.dataset.load('iris')

classes = data['class'].unique()
n_classes = len(classes)


fig, axes = plot.subplots(nrows=1, ncols=3)
for i, label in enumerate(classes):
    subdata = data.iloc[data['class']==label, 0:4]
    ax = axes[i]
    subdata.plot(ax=ax, marker='o', linestyle='None')
    ax.set_title(label)

plot.show()


