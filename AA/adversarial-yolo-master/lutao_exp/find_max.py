import cv2
import numpy as np

# 加载图像
image = cv2.imread('./ex3/8440.png', cv2.IMREAD_GRAYSCALE)

# 进行边缘检测
edges = cv2.Canny(image, 50, 150)

# 进行霍夫直线检测
lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)

# 提取横向和纵向两类直线
horizontal_lines = []
vertical_lines = []

for line in lines:
    rho, theta = line[0]
    angle = np.degrees(theta)

    # 将直线按角度分为横向和纵向两类
    if angle < 30 or angle > 150:
        horizontal_lines.append((rho, theta))
    elif angle > 60 and angle < 120:
        vertical_lines.append((rho, theta))

# 对横向和纵向直线按长度排序
horizontal_lines.sort(key=lambda x: x[0], reverse=True)
vertical_lines.sort(key=lambda x: x[0], reverse=True)

# 获取最长的两条横向和纵向直线
horizontal_lines = horizontal_lines[:2]
vertical_lines = vertical_lines[:2]

# 计算交点坐标
intersection_points = []
for h_line in horizontal_lines:
    for v_line in vertical_lines:
        rho_h, theta_h = h_line
        rho_v, theta_v = v_line

        # 计算交点坐标
        x = int((rho_v - rho_h) / np.cos(theta_h))
        y = int((rho_h / np.sin(theta_h) - rho_v / np.sin(theta_v)) / (1 / np.tan(theta_h) - 1 / np.tan(theta_v)))
        intersection_points.append((x, y))

print(intersection_points)

# import cv2
# import numpy as np
#
# # 加载图像
# image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)
#
# # 进行边缘检测
# edges = cv2.Canny(image, 50, 150)
#
# # 进行轮廓检测
# contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
# # 初始化最大包围框
# max_bounding_rect = None
#
# # 遍历轮廓
# for contour in contours:
#     # 计算轮廓的最小外接矩形
#     rect = cv2.minAreaRect(contour)
#     box = cv2.boxPoints(rect)
#     box = np.int0(box)
#
#     # 更新最大包围框
#     if max_bounding_rect is None:
#         max_bounding_rect = box
#     else:
#         max_bounding_rect[0][0] = min(max_bounding_rect[0][0], box[0][0])
#         max_bounding_rect[0][1] = min(max_bounding_rect[0][1], box[0][1])
#         max_bounding_rect[2][0] = max(max_bounding_rect[2][0], box[2][0])
#         max_bounding_rect[2][1] = max(max_bounding_rect[2][1], box[2][1])
#
# # 获取最大包围框的四个顶点坐标
# top_left = (max_bounding_rect[0][0], max_bounding_rect[0][1])
# top_right = (max_bounding_rect[1][0], max_bounding_rect[1][1])
# bottom_right = (max_bounding_rect[2][0], max_bounding_rect[2][1])
# bottom_left = (max_bounding_rect[3][0], max_bounding_rect[3][1])
#
# # 在图像中绘制最大包围框和交点
# cv2.drawContours(image, [max_bounding_rect], 0, (0, 255, 0), 2)
# cv2.circle(image, top_left, 5, (0, 0, 255), -1)
# cv2.circle(image, top_right, 5, (0, 0, 255), -1)
# cv2.circle(image, bottom_right, 5, (0, 0, 255), -1)
# cv2.circle(image, bottom_left, 5, (0, 0, 255), -1)
#
# # 显示图像
# cv2.imshow('Image with Maximum Bounding Rectangle and Intersection Points', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# # 打印最大包围框的四个顶点坐标
# print("Top Left: ({}, {})".format(top_left[0], top_left[1]))
# print("Top Right: ({}, {})".format(top_right[0], top_right[1]))
# print("Bottom Right: ({}, {})".format(bottom_right[0], bottom_right[1]))
