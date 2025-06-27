import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torchvision.utils import save_image
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import random
from torch.utils.data import DataLoader

def generate_custom_dataset(output_dir, num_samples_per_digit=20, digits=range(10)):
    """
    从MNIST数据集生成自定义数据集样本，保存为图片文件
    
    参数:
    output_dir -- 输出目录
    num_samples_per_digit -- 每个数字的样本数
    digits -- 要生成的数字列表
    """
    # 创建输出目录
    os.makedirs(os.path.join(output_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'val'), exist_ok=True)
    
    # 加载MNIST数据集
    mnist_train = datasets.MNIST(root='./data', train=True, download=True)
    
    # 为每个数字生成样本
    for digit in digits:
        # 找出数据集中指定数字的所有样本
        digit_indices = [i for i, (_, label) in enumerate(mnist_train) if label == digit]
        
        # 随机选择指定数量的样本
        if len(digit_indices) >= num_samples_per_digit:
            selected_indices = random.sample(digit_indices, num_samples_per_digit)
        else:
            selected_indices = digit_indices
            
        # 保存样本图片
        for i, idx in enumerate(selected_indices):
            # 分割为训练集和验证集
            if i < int(0.8 * len(selected_indices)):
                save_dir = os.path.join(output_dir, 'train')
            else:
                save_dir = os.path.join(output_dir, 'val')
                
            # 获取图像和标签
            image, label = mnist_train[idx]
            
            # 保存图像
            filename = f"{label}_{i+1:03d}.png"
            image.save(os.path.join(save_dir, filename))
    
    print(f"已生成自定义数据集，保存在 {output_dir}")
    print(f"训练样本: {len(os.listdir(os.path.join(output_dir, 'train')))} 张")
    print(f"验证样本: {len(os.listdir(os.path.join(output_dir, 'val')))} 张")

def augment_dataset(input_dir, output_dir, augmentation_factor=5):
    """
    对数据集进行数据增强，生成更多训练样本
    
    参数:
    input_dir -- 输入目录（原始图片）
    output_dir -- 输出目录（增强后图片）
    augmentation_factor -- 每张原始图片增强生成的新图片数量
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 复制原始图片到输出目录
    for filename in os.listdir(input_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_dir, filename)
            image = Image.open(image_path).convert('L')
            image.save(os.path.join(output_dir, filename))
    
    # 对每张图片进行数据增强
    for filename in os.listdir(input_dir):
        if not filename.endswith(('.png', '.jpg', '.jpeg')):
            continue
            
        image_path = os.path.join(input_dir, filename)
        image = Image.open(image_path).convert('L')
        label = filename.split('_')[0]
        base_name = filename.split('.')[0]
        
        for i in range(augmentation_factor):
            # 应用随机变换
            augmented = apply_random_transforms(image)
            
            # 保存增强后的图片
            aug_filename = f"{label}_{base_name}_aug{i+1}.png"
            augmented.save(os.path.join(output_dir, aug_filename))
    
    print(f"数据集增强完成，保存在 {output_dir}")
    print(f"样本数量: {len(os.listdir(output_dir))} 张")

def apply_random_transforms(image):
    """应用随机变换到图像"""
    # 复制原始图像
    img = image.copy()
    
    # 随机旋转 (-15到15度)
    if random.random() > 0.5:
        angle = random.uniform(-15, 15)
        img = img.rotate(angle, resample=Image.BILINEAR, expand=False)
    
    # 随机平移
    if random.random() > 0.5:
        width, height = img.size
        dx = int(random.uniform(-width*0.1, width*0.1))
        dy = int(random.uniform(-height*0.1, height*0.1))
        img = ImageOps.expand(img, border=(0, 0, -dx, -dy), fill=0)
        img = img.crop((0, 0, width, height))
    
    # 随机缩放
    if random.random() > 0.5:
        scale = random.uniform(0.8, 1.2)
        width, height = img.size
        new_width = int(width * scale)
        new_height = int(height * scale)
        img = img.resize((new_width, new_height), Image.BILINEAR)
        
        # 如果缩小了，需要添加边框；如果放大了，需要裁剪
        if scale < 1:
            result = Image.new('L', (width, height), 0)
            result.paste(img, ((width - new_width) // 2, (height - new_height) // 2))
            img = result
        else:
            left = (new_width - width) // 2
            top = (new_height - height) // 2
            img = img.crop((left, top, left + width, top + height))
    
    # 随机亮度调整
    if random.random() > 0.5:
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(random.uniform(0.7, 1.3))
    
    # 随机对比度调整
    if random.random() > 0.5:
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(random.uniform(0.7, 1.3))
    
    # 随机模糊
    if random.random() > 0.8:  # 较小概率应用模糊，避免过度模糊
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0, 1)))
    
    # 随机噪声
    if random.random() > 0.5:
        width, height = img.size
        noise = np.random.normal(0, 10, (height, width))
        img_array = np.array(img)
        noisy_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(noisy_array)
    
    return img

def visualize_dataset(data_dir, num_samples=10):
    """可视化数据集样本"""
    images = []
    for filename in os.listdir(data_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            images.append((os.path.join(data_dir, filename), filename))
            if len(images) >= num_samples:
                break
    
    # 创建图表
    num_rows = (len(images) + 4) // 5  # 每行最多5张图片
    fig, axes = plt.subplots(num_rows, min(5, len(images)), figsize=(12, 2*num_rows))
    if num_rows == 1:
        axes = [axes]
    
    # 展示图片
    for i, (image_path, filename) in enumerate(images):
        row, col = i // 5, i % 5
        ax = axes[row][col] if len(images) > 5 else axes[col]
        
        img = Image.open(image_path).convert('L')
        ax.imshow(img, cmap='gray')
        ax.set_title(f"{filename}")
        ax.axis('off')
    
    # 隐藏空白子图
    for i in range(len(images), num_rows * 5):
        row, col = i // 5, i % 5
        if col < min(5, len(images)):
            ax = axes[row][col] if len(images) > 5 else axes[col]
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('dataset_samples.png')
    plt.show()

def prepare_data_loaders(train_dir, val_dir, batch_size=32):
    """准备数据加载器"""
    transform_train = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    transform_test = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    from model import CustomHandwrittenDataset
    
    train_dataset = CustomHandwrittenDataset(train_dir, transform_train)
    val_dataset = CustomHandwrittenDataset(val_dir, transform_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

if __name__ == "__main__":
    # 示例用法
    # 1. 从MNIST生成初始数据集
    generate_custom_dataset('./custom_digits', num_samples_per_digit=30)
    
    # 2. 对训练集进行数据增强
    augment_dataset('./custom_digits/train', './custom_digits/train_augmented', augmentation_factor=3)
    
    # 3. 可视化增强后的数据集
    visualize_dataset('./custom_digits/train_augmented', num_samples=15) 