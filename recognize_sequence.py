import argparse
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import os
from model import load_model

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 预处理变换
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

def preprocess_image(image_path):
    """预处理图像：读取、灰度化、二值化等"""
    # 读取图像
    if isinstance(image_path, str):
        # 从文件路径读取
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    else:
        # 从PIL图像转换
        img = np.array(image_path.convert('L'))
    
    # 确保图像存在
    if img is None:
        raise ValueError(f"无法读取图像: {image_path}")
    
    # 调整大小，保持宽高比
    height, width = img.shape
    new_height = 100  # 固定高度
    new_width = int(width * (new_height / height))
    img = cv2.resize(img, (new_width, new_height))
    
    # 二值化处理
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 降噪
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    return img, binary

def segment_digits(binary):
    """分割连续数字为单个数字"""
    # 查找轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 根据x坐标排序轮廓（从左到右）
    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    
    # 过滤掉太小的轮廓（可能是噪声）
    min_area = 20  # 最小区域面积
    filtered_boxes = [(x, y, w, h) for (x, y, w, h) in bounding_boxes if w*h > min_area]
    
    # 按x坐标排序（从左到右）
    sorted_boxes = sorted(filtered_boxes, key=lambda x: x[0])
    
    return sorted_boxes

def segment_by_projection(binary):
    """使用投影方法分割数字（适用于数字挨得很近的情况）"""
    # 垂直投影
    v_projection = np.sum(binary, axis=0)
    
    # 找出数字之间的分隔点
    digit_regions = []
    in_digit = False
    start = 0
    
    # 阈值，低于此值视为分隔点
    threshold = max(v_projection) * 0.05
    
    for i, p in enumerate(v_projection):
        if p > threshold and not in_digit:
            # 数字开始
            in_digit = True
            start = i
        elif p <= threshold and in_digit:
            # 数字结束
            in_digit = False
            if i - start > 5:  # 过滤太窄的区域
                digit_regions.append((start, i))
    
    # 处理最后一个数字
    if in_digit:
        digit_regions.append((start, len(v_projection)))
    
    # 转换为边界框格式 (x, y, w, h)
    h = binary.shape[0]
    boxes = [(x, 0, x2-x, h) for x, x2 in digit_regions]
    
    return boxes

def extract_digits(image, binary, boxes, padding=2):
    """从原始图像中提取单个数字"""
    digits = []
    positions = []
    
    h, w = binary.shape
    
    for i, (x, y, w, h) in enumerate(boxes):
        # 添加一些边距
        x_start = max(0, x - padding)
        y_start = max(0, y - padding)
        x_end = min(binary.shape[1], x + w + padding)
        y_end = min(binary.shape[0], y + h + padding)
        
        # 提取数字区域
        digit = binary[y_start:y_end, x_start:x_end]
        
        # 确保提取的区域不为空
        if digit.size == 0:
            continue
        
        # 计算边距，使数字居中
        top = bottom = left = right = 0
        aspect_ratio = digit.shape[1] / digit.shape[0]
        
        # 处理宽高比，使数字更像MNIST格式
        if aspect_ratio > 1:  # 宽>高
            # 添加上下边距使之成为正方形
            diff = int((digit.shape[1] - digit.shape[0]) / 2)
            top = bottom = diff
        else:  # 高>宽
            # 添加左右边距使之成为正方形
            diff = int((digit.shape[0] - digit.shape[1]) / 2)
            left = right = diff
        
        # 添加边距
        digit_padded = cv2.copyMakeBorder(
            digit, top, bottom, left, right, 
            cv2.BORDER_CONSTANT, value=0
        )
        
        # 调整大小为28x28以匹配MNIST
        digit_resized = cv2.resize(digit_padded, (28, 28))
        
        # 转换为PIL图像
        digit_pil = Image.fromarray(digit_resized)
        
        digits.append(digit_pil)
        positions.append((x, x+w))  # 记录位置用于排序
    
    # 按位置从左到右排序
    sorted_indices = sorted(range(len(positions)), key=lambda i: positions[i][0])
    sorted_digits = [digits[i] for i in sorted_indices]
    
    return sorted_digits

def recognize_sequence(image_path, model_path='models/finetuned_model.pth', visualize=False):
    """识别图像中的连续数字序列"""
    # 加载模型
    print(f"加载模型: {model_path}")
    model, _ = load_model(model_path, device)
    model.eval()
    
    # 预处理图像
    print("预处理图像...")
    original, binary = preprocess_image(image_path)
    
    # 尝试两种分割方法
    boxes = segment_digits(binary)
    
    # 如果常规分割找不到足够的数字，尝试投影分割
    if len(boxes) < 2:
        boxes = segment_by_projection(binary)
    
    # 提取单个数字
    print(f"分割出 {len(boxes)} 个数字...")
    digit_images = extract_digits(original, binary, boxes)
    
    if len(digit_images) == 0:
        print("未能分割出任何数字，请尝试调整图像或手动分割。")
        return ""
    
    # 识别每个数字
    print("识别数字...")
    predictions = []
    confidences = []
    
    for digit_img in digit_images:
        # 转换为模型输入格式
        digit_tensor = transform(digit_img).unsqueeze(0).to(device)
        
        # 进行预测
        with torch.no_grad():
            output = model(digit_tensor)
            probs = torch.nn.functional.softmax(output, dim=1)
            conf, pred = torch.max(probs, 1)
        
        predictions.append(pred.item())
        confidences.append(conf.item())
    
    # 组合结果
    sequence = ''.join(map(str, predictions))
    
    # 输出结果
    print(f"识别结果: {sequence}")
    print(f"平均置信度: {sum(confidences)/len(confidences):.4f}")
    
    # 可视化
    if visualize:
        visualize_results(original, binary, boxes, digit_images, predictions, confidences)
    
    return sequence

def visualize_results(original, binary, boxes, digit_images, predictions, confidences):
    """可视化分割和识别结果"""
    # 创建一个包含3行的图
    fig = plt.figure(figsize=(15, 10))
    
    # 1. 原始图像和分割框
    ax1 = fig.add_subplot(3, 1, 1)
    ax1.imshow(original, cmap='gray')
    ax1.set_title('原始图像和分割')
    
    # 在原始图像上画出分割框
    img_with_boxes = cv2.cvtColor(original, cv2.COLOR_GRAY2RGB)
    for x, y, w, h in boxes:
        cv2.rectangle(img_with_boxes, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    ax1.imshow(img_with_boxes)
    ax1.axis('off')
    
    # 2. 二值化图像
    ax2 = fig.add_subplot(3, 1, 2)
    ax2.imshow(binary, cmap='gray')
    ax2.set_title('二值化图像')
    ax2.axis('off')
    
    # 3. 分割后的数字及识别结果
    ax3 = fig.add_subplot(3, 1, 3)
    ax3.set_title('分割后的数字和识别结果')
    ax3.axis('off')
    
    # 排列分割后的数字
    num_digits = len(digit_images)
    for i, (digit_img, pred, conf) in enumerate(zip(digit_images, predictions, confidences)):
        # 在子图中添加每个数字
        ax = fig.add_subplot(3, num_digits, 2*num_digits + i + 1)
        ax.imshow(digit_img, cmap='gray')
        ax.set_title(f'预测: {pred}\n置信度: {conf:.2f}')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('sequence_recognition_result.png')
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='手写学号（连续数字）识别')
    parser.add_argument('--image', type=str, required=True, help='要识别的图像路径')
    parser.add_argument('--model', type=str, default='models/finetuned_model.pth', help='模型文件路径')
    parser.add_argument('--visualize', action='store_true', help='是否可视化结果')
    
    args = parser.parse_args()
    
    # 执行识别
    recognize_sequence(args.image, args.model, args.visualize)

if __name__ == "__main__":
    main() 