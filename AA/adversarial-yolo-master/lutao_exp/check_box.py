import cv2
import numpy as np

# 加载图像
image = cv2.imread('./ex3/8440.png', cv2.IMREAD_GRAYSCALE)

# 进行边缘检测
edges = cv2.Canny(image, 50, 150)

# 进行轮廓检测
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 遍历轮廓
rectangles = []
for contour in contours:
    # 计算轮廓的近似多边形
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    # 如果近似多边形有四个顶点，则认为是矩形
    if len(approx) == 4:
        rectangles.append(approx)

# 获取矩形的交点坐标
intersection_points = []
for rect in rectangles:
    for i in range(4):
        for j in range(i + 1, 4):
            pt1 = rect[i][0]
            pt2 = rect[j][0]
            intersection_points.append((pt1[0], pt1[1]))
            intersection_points.append((pt2[0], pt2[1]))

# 显示图像并绘制矩形和交点
for rect in rectangles:
    cv2.drawContours(image, [rect], 0, (0, 255, 0), 2)
for pt in intersection_points:
    cv2.circle(image, pt, 5, (0, 0, 255), -1)

# # 显示图像
# cv2.imshow('Image with Rectangles and Intersection Points', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# 返回交点的四个顶点坐标
print("Intersection Points:")
for i, pt in enumerate(intersection_points):
    print("Point {}: ({}, {})".format(i + 1, pt[0], pt[1]))
