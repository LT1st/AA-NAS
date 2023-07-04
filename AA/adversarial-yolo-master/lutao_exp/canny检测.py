import cv2
import matplotlib.pyplot as plt
import numpy as np

path = './batchx.png'
path = './ex3/8440.png'

# 读取图像
image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # 替换为你的图像路径，并将其转换为灰度图像

# 进行Canny边缘检测
edges = cv2.Canny(image, 50, 150)  # 参数50和150分别为低阈值和高阈值

# 显示原始图像
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

# 显示边缘图像
plt.subplot(1, 2, 2)
plt.imshow(edges, cmap='gray')
plt.title('Canny Edges')
plt.axis('off')

plt.show()

# 统计频率分布
frequencies, _ = np.histogram(edges, bins=range(256))
plt.bar(range(256), frequencies)
plt.xlabel('Intensity')
plt.ylabel('Frequency')
plt.title('Frequency Distribution of Canny Edges')
plt.show()
