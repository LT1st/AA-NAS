import cv2
import screeninfo
import numpy as np
import random


def display_image_on_secondary_monitor(image, monitor_id=0):


    # 获取所有显示器的信息
    monitors = screeninfo.get_monitors()

    # 检查是否至少有两个显示器
    if len(monitors) < 2:
        print("无法找到第二个显示器。")
        return

    # 获取第二个显示器的信息
    second_monitor = monitors[monitor_id]

    # 创建一个窗口并在第二个显示器上显示图像
    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.moveWindow("Image", second_monitor.x, second_monitor.y)  # 将窗口移动到第二个显示器的位置
    cv2.setWindowProperty("Image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def generate_random_image(width, height):
    # 生成随机的像素值
    random_pixels = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)

    # 创建图像
    image = np.zeros((height, width, 3), dtype=np.uint8)

    # 填充图像的像素值
    for y in range(height):
        for x in range(width):
            # 设置随机颜色
            image[y, x] = random_pixels[y, x]

    return image


if __name__ == '__main__':
    # 指定图像的宽度和高度
    width = 800
    height = 600

    # 生成随机图像
    image = generate_random_image(width, height)


    # image_path = "path/to/your/image.jpg"
    # # 读取图像
    # image = cv2.imread(image_path)
    display_image_on_secondary_monitor(image)