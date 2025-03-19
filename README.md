## RT-DETR for SAR Ship Detection

RTDETR-SAR 是一个用于SAR（合成孔径雷达）图像中舰船目标检测的深度学习项目，基于RT-DETR（Real-Time Detection Transformer）模型构建。以下是项目的结构分析：

### 1 整体架构
项目采用模块化设计，主要包含以下几个部分：

- src 目录：核心源代码

  - core：包含基础组件，如注册器（register）
  - data：数据加载和处理
  - coco：基于COCO格式的数据集实现，主要使用 HRSID 舰船数据集
  - misc：辅助工具，如 logger, visualizer
  - nn：神经网络模块
    - backbone：主干网络，包括 ResNet, DLA 等
  - solver：训练和评估引擎
    - det_engine.py：训练和评估的核心函数
    - det_solver.py：封装训练评估流程的求解器类
  - zoo：模型库
    - rtdetr：RT-DETR模型的实现，包括 hybrid_encoder 等
- data 目录：数据集

  HRSID：高分辨率SAR船只检测数据集
annotations：COCO格式的标注文件
train：训练图像
val：验证图像
dataset_tools.ipynb：数据集处理工具
- tools 目录：工具脚本

  - train.py：训练脚本

  - export_onnx.py：模型导出为ONNX格式的工具

  - README.md：使用说明

- configs 目录：配置文件
  - dataset: COCO 数据集和 HRSID 数据集的配置文件
  - rtdetr：RTDETR 模型相关配置

### 2 关键组件
1. 模型架构：项目使用了多种深度学习模型架构，主要包括：

    - RT-DETR：一种实时的检测Transformer
    - 各种主干网络（如DLA, ResNet等）
    - HybridEncoder：混合编码器，包含卷积和Transformer组件

2. 数据处理：
    - 基于 COCO 格式定制的 HRSID 数据集加载器
    - 数据增强和预处理
    - 专门处理SAR图像中的舰船目标
3. 训练与评估
    - 支持单GP U和多GPU 训练
    - 支持混合精度训练（AMP）
    - 支持 EMA（指数移动平均）模型更新
    - 基于 COCO 评估器的性能评估
    - 支持检查点保存和恢复
4. 模型导出
    - 支持将训练好的模型导出为ONNX格式
    - 支持模型简化和验证


### 3 Quick start

<details>
<summary>Install</summary>

```bash
pip install -r requirements.txt
```

</details>


<details>
<summary>Data</summary>

- Download and extract HRSID train and val images.
```
data/HRSID/
  annotations/  # annotation json files
  train/    # train images
  val/      # val images
```
- Modify config `img_folder`, `ann_file`
</details>



<details>
<summary>Training & Evaluation</summary>

- Training on a Single GPU:

```shell
# training on single-gpu
export CUDA_VISIBLE_DEVICES=0
python tools/train.py -c configs/rtdetr/rtdetr_r50vd_6x_hrsid.yml
```

- Training on Multiple GPUs:

```shell
# train on multi-gpu
export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc_per_node=4 tools/train.py -c configs/rtdetr/rtdetr_r50vd_6x_hrsid.yml
```

- Evaluation on Multiple GPUs:

```shell
# val on multi-gpu
export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc_per_node=4 tools/train.py -c configs/rtdetr/rtdetr_r50vd_6x_hrsid.yml -r path/to/checkpoint --test-only
```

</details>



<details>
<summary>Export</summary>

```shell
python tools/export_onnx.py -c configs/rtdetr/rtdetr_r18vd_6x_hrsid.yml -r path/to/checkpoint --check
```
</details>






