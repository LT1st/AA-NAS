import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

# Load the pre-trained model and remove the last layer
model = models.resnet18(pretrained=True)
model = nn.Sequential(*list(model.children())[:-1])  # remove last layer
model.eval()  # set the model to evaluation mode

# Move model to device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

def extract_features(img):
    # Assume img is a BCHW tensor
    img = img.to(device)  # move image to device
    with torch.no_grad():
        features = model(img)  # extract features
    return features

def resize_features(features, size=( 1, 1)):
    # Resize features to a fixed size
    features = F.interpolate(features, size=size, mode='bilinear', align_corners=False)
    return features

# Assume img is your input image with shape BCHW
# img = ...
# 生成随机BCHW向量
batch_size = 4
channels = 3
height = 256
width = 256
img = torch.randn(batch_size, channels, height, width)

# 4 512 1 1
# Extract features
features = extract_features(img)

# Resize features
features_resized = resize_features(features)

print(features_resized.shape)  # Should be [B, 1, 100, 100]