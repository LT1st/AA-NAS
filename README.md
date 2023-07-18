# AA-NAS
See English Version: [readme in English](./markdown_EN.md)
## 文件结构
| Folder/File Name | Description |
|------------------|-------------|
| scripts/         | shell启动脚本   |
| main.py          | Python启动脚本  |
| testing/         | AA数据        |
| data/            | NAS训练数据集    |
| weights/         | 权重          |
| NAS/             | 用于nas的代码    |   
| AA/              | 用于AA的代码     |

# TODO List
- [x] 7.4 解决文件结构变化导致的BUG
- [x] 7.7 解决FNA的环境适配
- [ ] deformable思路的替换，使用交替训练
- [ ] 7,12 分类模型对不上去， 数据集解压出问题

## 数据处理
- [ ]  用于NAS的coco数据集需要保证不被污染，考虑在使用前利用脚本清理生成的数据
- [ ]  数据冗余较高，训练过程中动态清除？手动清除？
- [ ]  coco软连接时，怎么操作数据集？

## AA using patches
- [ ]  多模型deformable
- [ ]  比较循环训练和deformable
- [x] 包路径问题，import搜索不到，chdir安全，`sys.path.append()`可能yinport不到正确的包

## NAS for OD
- [ ]  muti-shot 搜素太慢  
- [ ]  zero-shot 如何把数据加进去？ https://github.com/alibaba/lightweight-neural-architecture-search/blob/main/installation.md
- [x]  one-shot 最合适 DetNAS https://github.com/LT1st/DetNAS

### 开源复现目标
| 论文      | 特点           | 时间   | 评价               | 地址                                                                        |
|-----------|----------------|------|------------------|-----------------------------------------------------------------------------|
| tiny NAS  | Training-Free  | 6小时  | 零样本，加入AE需要调整优化目标 | https://github.com/alibaba/lightweight-neural-architecture-search/tree/main |
| NAS-FCOS  |                |      |                  |                                                                             |
| NAS-FPN   |                | 32天  | 早期工作，太慢          | https://github.com/LT1st/NAS_FPN_Tensorflow                                 |
| NEAS      |                |      |                  | https://github.com/LT1st/NEAS                                               |
| FNA       |                | 3-4天 | 网络进化的思路，好        | https://github.com/LT1st/FNA                                                |
| ZenNAS    |       Training-Free         |      |                  | https://github.com/LT1st/ZenNAS                                             |
| HITDET    |                |      |                  | https://github.com/LT1st/HitDet.pytorch                                     |
| OPANAS    |                |      |                  | https://github.com/LT1st/OPANAS                                             |

## 接口
在main.py中控制参数，后期再优化流程。