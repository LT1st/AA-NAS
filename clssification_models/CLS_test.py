import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os

# Load the pre-trained model
model_path = '/path/to/your/model.pth'
model = torch.load(model_path)
model.eval()

# Define the data transforms for input images
transform = transforms.Compose([
    transforms.Resize(32),
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2762])
])

# Load the test dataset
test_dir = '/path/to/your/test/images'
test_dataset = datasets.ImageFolder(test_dir, transform=transform)

# Define the categories to count for top-1 and top-5 accuracy
human_categories = [i for i in range(80, 100)] # 80-99 are human categories

# Initialize counters for top-1 and top-5 accuracy
top1_count = 0
top5_count = 0
total_count = 0

# Loop over the test dataset and perform inference
for i, (image, label) in enumerate(test_dataset):
    # Perform inference on the image
    output = model(image.unsqueeze(0))
    _, predicted = torch.max(output.data, 1)
    _, top5 = torch.topk(output.data, 5)

    # Update the counters for top-1 and top-5 accuracy
    if predicted == label:
        top1_count += 1
    if label in top5:
        top5_count += 1
    if label in human_categories:
        total_count += 1

# Compute and print the accuracy results
top1_acc = top1_count / len(test_dataset)
top5_acc = top5_count / len(test_dataset)
human_freq = total_count / len(test_dataset)
print(f'Top-1 accuracy: {top1_acc:.4f}')
print(f'Top-5 accuracy: {top5_acc:.4f}')
print(f'Human category frequency: {human_freq:.4f}')