# AA-NAS
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
-[ ] 7.4 解决文件结构变化导致的BUG

## 数据处理
- [ ]  用于NAS的coco数据集需要保证不被污染，考虑在使用前利用脚本清理生成的数据
- [ ]  数据冗余较高，训练过程中动态清除？手动清除？
- [ ]  coco软连接时，怎么操作数据集？

## AA using patches
- [ ]  多模型deformable
- [ ]  比较循环训练和deformable

## NAS for OD
- [ ]  muti-shot 搜素太慢  
- [ ]  zero-shot 如何把数据加进去？ https://github.com/alibaba/lightweight-neural-architecture-search/blob/main/installation.md
- [x]  one-shot 最合适 DetNAS https://github.com/LT1st/DetNAS