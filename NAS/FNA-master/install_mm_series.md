# For all 

# For FNA

* cuda 10.1 (Or the mmdet will not able to install)
* python 3.7
* pytorch 1.1  https://pytorch.org/get-started/previous-versions/#v110
* mmdet 0.6.0 (53c647e)
* mmcv 0.2.10  # 可能需要从源码编译 https://mmcv.readthedocs.io/zh_CN/latest/get_started/build.html
* Pillow 未知版本
* chardet
* torchvision

Check CUDA version：
```
import torch
import subprocess
import os


def get_cuda_version():
    try:
        return torch.version.cuda
    except:
        return None

def check_cuda_version():
    result = subprocess.run(['nvcc', '--version'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    for line in result.stdout.decode().split('\n'):
        if 'release' in line:
            return line.strip()
    return None

def is_cuda_available():
    return torch.cuda.is_available()

get_cuda_version()
check_cuda_version()
is_cuda_available()
for key, value in os.environ.items():
    if 'cuda' in key.lower() or 'cudnn' in key.lower():
        print(f'{key}: {value}')


```
pytorch 没有合适版本
```conda install pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=10.0 -c pytorch```
查找版本
```
import requests
from bs4 import BeautifulSoup

tep = 'torch'
url = f"https://pypi.org/project/{tep}/#history"

page = requests.get(url)
soup = BeautifulSoup(page.content, 'html.parser')
version_tags = soup.find_all('span', {'class': 'release__version'})

versions = [tag.get_text() for tag in version_tags]

print(f'Available versions of {tep} on PyPI: {versions}')
```

Check your version:
```
import platform
import torch

print(f'System type: {torch.cuda.get_device_properties(0).name}')
print(f'CUDA version: {torch.version.cuda}')
print(f'PyTorch version: {torch.__version__}')
print(f'Operating system: {platform.system()}')
```
## mmcv
https://zhuanlan.zhihu.com/p/434491590
配置 vs 2022 编译环境
`C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.29.30133\bin\HostX86\x64`


The required version is under `1.0.0`, which means the `mmcv-full` is not taken into consideration.
You can see more detail at : https://mmcv.readthedocs.io/zh_CN/latest/get_started/installation.html