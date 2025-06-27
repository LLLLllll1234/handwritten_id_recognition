# 手写学号识别系统

这个系统使用PyTorch实现了一个可以识别手写学号的深度学习模型。该系统先在MNIST数据集上进行预训练，然后可以在自定义的手写学号数据集上进行微调，从而实现对手写学号的准确识别。

## 项目结构

```
.
├── main.py                # 主程序（训练和微调模型）
├── model.py               # 模型定义文件
├── predict.py             # 预测脚本
├── data_utils.py          # 数据处理工具
├── generate_dataset.py    # 数据集生成脚本
├── custom_digits/         # 自定义数据集目录
│   ├── train/             # 训练集
│   ├── train_augmented/   # 增强后的训练集
│   └── val/               # 验证集
└── models/                # 保存训练好的模型
    ├── mnist_pretrained_model.pth  # MNIST预训练模型
    └── finetuned_model.pth         # 微调后的模型
```

手动创建虚拟环境

## 环境要求

- Python 3.6+
- PyTorch 1.7+
- torchvision
- matplotlib
- numpy
- PIL

可以通过以下命令安装依赖：

```bash
pip install torch torchvision numpy matplotlib pillow
```

## 使用步骤

### 1. 准备数据集

您可以使用提供的脚本从MNIST数据集生成初始数据集，然后进行数据增强：

```bash
# 生成基础数据集
python generate_dataset.py

# 生成数据集并进行数据增强
python generate_dataset.py --augment --augment_factor 3

# 生成数据集、进行数据增强并可视化
python generate_dataset.py --augment --visualize
```

如果您有自己的手写学号图片，可以直接将其放入相应目录：

- 训练图片：`custom_digits/train/`
- 验证图片：`custom_digits/val/`

图片命名格式应为 `{数字}_{序号}.png` （例如：`0_001.png`，`1_002.png`等）。

### 2. 模型训练

#### 2.1 在MNIST上预训练

```bash
python main.py
```

这将加载MNIST数据集，训练模型并保存预训练模型到`models/mnist_pretrained_model.pth`。

#### 2.2 在自定义数据集上微调

确保您已经准备好了自定义数据集，然后运行：

```bash
python main.py --finetune
```

微调后的模型将保存到`models/finetuned_model.pth`。

### 3. 预测识别

使用训练好的模型对手写数字图片进行预测：

```bash
# 预测单张图片
python predict.py --image path/to/image.png --visualize

# 批量预测文件夹中的图片
python predict.py --dir path/to/images/

# 使用特定模型进行预测
python predict.py --model models/finetuned_model.pth --image path/to/image.png
```

## 模型架构

该项目使用了一个多层CNN模型，具有以下特点：

- 3个卷积层，每层后接批归一化和ReLU激活函数
- 池化层用于降低特征图尺寸
- Dropout层用于防止过拟合
- 3个全连接层进行最终分类

## 数据增强方法

为了提高模型的泛化能力，系统提供了多种数据增强方法：

- 随机旋转（-15°~15°）
- 随机平移
- 随机缩放
- 随机调整亮度和对比度
- 随机高斯模糊
- 随机噪声添加

这些方法可以帮助模型更好地适应各种手写风格和图像质量。

## 性能评估

模型评估指标包括：

- 总体准确率
- 各数字类别的准确率
- 训练和验证损失曲线
- 训练和验证准确率曲线

## 注意事项

- 如果您使用自己的手写图片，请确保将图片转换为灰度模式，并按照合适的命名规则组织。
- 为了提高识别准确率，建议提供足够数量的每个数字的样本（至少10个以上）。
- 如果识别效果不佳，可以尝试增加数据增强的强度或样本数量。 

##添加分割脚本recognize_sequence.py，可以识别连续手写数字（学号）
- 该脚本：
预处理图像（调整大小、二值化、降噪）
使用两种方法分割连续数字：
基于轮廓的分割（适用于分离较好的数字）
基于投影的分割（适用于数字挨得很近的情况）
提取每个单独的数字并调整为模型可接受的格式
使用训练好的模型识别每个数字
组合结果生成完整的学号

# 基本使用
python recognize_sequence.py --image path/to/your/id_image.jpg

# 带可视化
python recognize_sequence.py --image path/to/your/id_image.jpg --visualize

# 指定模型
python recognize_sequence.py --image path/to/your/id_image.jpg --model models/your_model.pth
