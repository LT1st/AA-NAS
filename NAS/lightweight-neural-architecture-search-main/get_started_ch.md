对于初次使用，请按照以下步骤顺序进行。

## 查找模型架构

- 目录[configs/classification](configs/classification)和[configs/detection](configs/detection)分别包含分类和检测的搜索配置文件，如下所示。
- 以分类任务为例，查找ResNet-50的CNN架构并执行以下步骤：

    ```shell
    sh tools/dist_search.sh configs/classification/R50_FLOPs.py
    # 或者
    python tools/search.py configs/classification/R50_FLOPs.py
  
  python tools/search.py configs/damoyolo/damoyolo_k1kx_small.py
  python tools/search.py configs/damoyolo/damoyolo_k1kx_tiny.py
  
    ```
### bug
python 版本需要大于等于3.9
需要安装库
modelscope

## 导出搜索结果

- 提供了一个导出搜索模型架构及其相关依赖项的脚本[tools/export.py](tools/export.py)，以便快速验证演示效果。

- 以[R50_FLOPs](configs/classification/R50_FLOPs.py)为例：

    ```shell
    python tools/export.py save_model/R50_R224_FLOPs41e8 output_dir
    ```

    将演示部署和相关代码复制到**output_dir/R50_R224_FLOPs41e8/**目录下，应包括以下内容：

    - best_structure.json：在搜索过程中找到的几个最佳模型架构。
    - demo.py：演示如何使用模型的简单脚本。
    - cnnnet.py：用于构建模型的类定义和实用函数。
    - modules：模型的基础模块。
    - weights/：搜索过程中找到的几个最佳模型权重（仅适用于一次性NAS方法）。

## 使用搜索的架构

- [demo.py](tinynas/deploy/cnnnet/demo.py)是一个基本的使用示例，但在上一步导出模型架构后，你也可以直接运行demo.py。

- 以用于分类任务的ResNet-50架构为例，下面解释核心代码：

    - 导入依赖项

    ```python
    import ast
    from cnnnet import CnnNet
    ```

    - 从文件中加载最佳结构。

    ```python
    with open('best_structure.json', 'r') as fin:
        content = fin.read()
        output_structures = ast.literal_eval(content)

    network_arch = output_structures['space_arch']
    best_structures = output_structures['best_structures']
    ```

    - 实例化分类主干网络。

    ```python
    network_id = 0    # 索引号。在example_cls_res50.sh中设置num_network=5，可以输出多个结构。
    out_indices = (4, )    # 输出阶段。对于分类任务，只需获取最后一阶段的输出。

    backbone = CnnNet(
        structure_info=best_structures[network_id],
        out_indices=out_indices,
        num_classes=1000,
        classification=True,
        )
    backbone.init_weight(pretrained)
    ```

    - 现在可以充分利用`backbone` :smile:

- 关于CNN检测任务模型的进一步使用方法，请参见[tinynas/deploy/]。