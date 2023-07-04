# Basic imports
import brambox as bb
import numpy as np
import pandas as pd
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns

# Settings
sns.set(style='darkgrid', context='notebook')  # Nicer plotting colors
# bb.logger.setConsoleLevel('ERROR')             # Only show error log messages

# annotations = bb.io.load(fmt='anno_darknet', path='inria/Test/pos/yolo-labels/',
#                          class_label_map={0: 'person'}, image_dims=lambda x: (1, 1))
annotations = bb.io.load(fmt='pascalvoc', path='inria/Test/pos/yolo-labels/',
                         class_label_map={0: 'person'})