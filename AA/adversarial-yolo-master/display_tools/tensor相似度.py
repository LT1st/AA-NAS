import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")