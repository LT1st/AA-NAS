import os

def check_file_size(path):

    ifToooooLarage = False
    for root, dirs, files in os.walk(path):
        for filename in files:
            filepath = os.path.join(root, filename)
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            if size_mb > 100:
                ifToooooLarage = True
                print(f"File size is {size_mb:.2f} MB: {filepath}")
    if not ifToooooLarage:
        print("able to publish")

# 检查当前目录下所有文件
check_file_size(".")