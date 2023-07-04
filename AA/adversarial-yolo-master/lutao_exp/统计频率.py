import cv2
import numpy as np
import matplotlib.pyplot as plt

# 加载图像
path = './ex3/8440.png'
path = './batchx.png'
image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

# 提取图像块
block_size = 4
blocks = []
for y in range(0, image.shape[0], block_size):
    for x in range(0, image.shape[1], block_size):
        block = image[y:y+block_size, x:x+block_size]
        blocks.append(block)

# 计算图像块的频率
frequencies = []
for block in blocks:
    frequency = np.mean(block)
    frequencies.append(frequency)

# 统计频率分布
plt.hist(frequencies, bins=range(256))
plt.xlabel('Frequency')
plt.ylabel('Count')
plt.title('Frequency Distribution of 8x8 Blocks')
plt.show()

# 分析频率分布
# 观察频率分布图，检查是否存在明显的峰值或集中区域，如果存在明显的频率集中，且频率值接近于某个特定值，
# 可以认为图像中存在8x8像素的规律方格。
