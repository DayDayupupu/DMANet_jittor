# AAAI_Event_based_detection——jittor框架复现

## 一、摘要

本项目是依据Dual Memory Aggregation Network for Event-based Object Detection with Learnable Representation这篇论文的开源项目在jittor框架下复现的结果，以及对其存在的问题做出来简单的总结与修改。由于计算资源有限，未能完全下载数据集（268G）进行彻底的复现该论文，而是选取了部分数据集在pytorch和jittor框架下分别实验，实验结果与pytorch框架复现基本一致。

## 二、如何运行

### Step1. 配置环境

```Python
git clone https://github.com/DayDayupupu/DFMNet_jittor.git
cd DMANet_jittor
conda create -n dmanet python=3.9.21
conda activate dmanet
conda install jittor==1.3.7.0 
pip install -r requirements.txt
```

### step2. Data preparation

具体数据处理过程可以参考原论文，此处是使用处理好的Prophesee 数据集

- Download: Mpx Auto-Detection Sub Dataset. (Total 268GB)

Links:[ https://pan.baidu.com/s/1YawxZFJhQWVgLye9zZtysA]( https://pan.baidu.com/s/1YawxZFJhQWVgLye9zZtysA)

Password: c6j9

- Dataset structure

```Python
prophesee_dlut   
├── test
│   ├── testfilelist00
├── train
│   ├── trainfilelist00
│   ├── trainfilelist01
└── val
    ├── valfilelist00
```

- Dataset Visualization

```Python
python data_check_npz.py
```

### step3. Training & Testing

首先添加log（记录训练曲线）、save path（保存测试结果）两个文件夹

Training

```Python
python train_DMANet.py 
```

Testing

```Python
python test_dmanet.py --weight=$YOUR_MODEL_PATH
```

## 三、复现结果

### 1. jittor

通过tensorboard记录训练曲线

#### learning rate

![image](https://github.com/user-attachments/assets/dce49519-6ce8-4452-8269-a1c82d0dfe3b)


#### training loss

![image](https://github.com/user-attachments/assets/59494ceb-9504-4531-921c-ca9083382567)


#### test result

数据集的较小，与训练集同源缺乏多样性，所以得分较高，主要是为了测试复现过程是否正确，方便与pytorch的框架进行对齐

|Class|Events|Labels|Precision|Recall|mAP@0.5|mAP@0.5:0.95|
|-|-|-|-|-|-|-|
|all|30|150|0.743|0.578|0.711|0.385|
|car|30|150|0.743|0.578|0.711|0.385|

### 2. pytorch

#### learning rate

![image](https://github.com/user-attachments/assets/4645b3e9-60c1-41aa-9e63-e1519bc8a4f7)


#### training loss

![image](https://github.com/user-attachments/assets/e5722c09-58ab-4e56-85c8-d5443d678f3d)


#### test result

|Class|Events|Labels|Precision|Recall|mAP@0.5|mAP@0.5:0.95|
|-|-|-|-|-|-|-|
|all|60|309|1.000|0.773|0.778|0.572|
|car|60|309|1.000|0.773|0.778|0.572|

## 四、可视化

tools中存在可视化工具，可以检查测试后的结果——prediction_visualize_npz.py（需修改路径）

![image](https://github.com/user-attachments/assets/cffb1003-45ac-4c2c-8b14-551087c9543e)


## 五、具体复现过程

参考
[复现过程笔记.md](https://github.com/DayDayupupu/DFMNet_jittor/blob/main/%E5%A4%8D%E7%8E%B0%E8%BF%87%E7%A8%8B%E7%AC%94%E8%AE%B0.md)（部分公式无法正常显示）
或者
https://flowus.cn/biaoguo/share/d1319969-d9f1-4a1c-8bd3-caace8b8aad4?code=ZMTFPB

## 六、原项目地址以及相关仓库

原项目地址

- AAAI_Event_based_detection：[https://github.com/wds320/AAAI_Event_based_detection](https://github.com/wds320/AAAI_Event_based_detection)

相关仓库

- RetinaNet implementation: [https://github.com/yhenon/pytorch-retinanet](https://github.com/yhenon/pytorch-retinanet)

- PointPillars implementation: [https://github.com/SmallMunich/nutonomy_pointpillars](https://github.com/SmallMunich/nutonomy_pointpillars)

- Prophesee's Automotive Dataset Toolbox: [https://github.com/prophesee-ai/prophesee-automotive-dataset-toolbox](https://github.com/prophesee-ai/prophesee-automotive-dataset-toolbox)

- Event-based Asynchronous Sparse Convolutional Networks: [https://github.com/uzh-rpg/rpg_asynet](https://github.com/uzh-rpg/rpg_asynet)

