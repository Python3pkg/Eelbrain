# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from eelbrain import *

src = datasets.get_mne_sample(src='ico', sub=[0])['src']
brain = plot.brain.brain(src.source, mask=False, hemi='lh', views='lat')
