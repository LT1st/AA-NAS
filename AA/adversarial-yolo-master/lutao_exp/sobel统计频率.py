import cv2
import numpy as np
import matplotlib.pyplot as plt

path = './batchx.png'
path = './ex3/8440.png'

image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=7)
sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=7)

# 将图像从BGR格式转换为RGB格式
image = cv2.cvtColor(sobely, cv2.COLOR_BGR2RGB)

# 显示图像
plt.imshow(image)
plt.axis('off')  # 关闭坐标轴
plt.show()

# gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
# frequencies, _ = np.histogram(gradient_magnitude.flatten(), bins=range(256))
# plt.hist(frequencies, bins=range(100))
# plt.xlabel('Frequency')
# plt.ylabel('Count')
# plt.title('Frequency Distribution')
# plt.show()
