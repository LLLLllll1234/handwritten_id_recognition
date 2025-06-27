import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from model import load_model

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 预处理变换
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

def predict_digit(model, image_path):
    """预测单张图片的数字"""
    image = Image.open(image_path).convert('L')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = F.softmax(output, dim=1)
        pred_prob, predicted = torch.max(probabilities, 1)
    
    return predicted.item(), pred_prob.item(), probabilities.cpu().numpy()[0]

def visualize_prediction(image_path, predicted, probabilities):
    """可视化预测结果"""
    image = Image.open(image_path).convert('L')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    # 显示图像
    ax1.imshow(image, cmap='gray')
    ax1.set_title(f'预测结果: {predicted}')
    ax1.axis('off')
    
    # 显示各类别概率
    bars = ax2.bar(np.arange(10), probabilities)
    bars[predicted].set_color('red')
    ax2.set_xticks(np.arange(10))
    ax2.set_xlabel('数字类别')
    ax2.set_ylabel('概率')
    ax2.set_title('预测概率分布')
    ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('prediction_result.png')
    plt.show()

def batch_predict(model, image_dir):
    """批量预测文件夹中的所有图片"""
    results = []
    
    for filename in os.listdir(image_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(image_dir, filename)
            pred_digit, confidence, _ = predict_digit(model, image_path)
            
            # 尝试从文件名获取真实标签
            try:
                true_label = int(filename.split('_')[0])
                correct = pred_digit == true_label
            except:
                true_label = "未知"
                correct = None
            
            results.append({
                'filename': filename,
                'predicted': pred_digit,
                'true_label': true_label,
                'confidence': confidence,
                'correct': correct
            })
            
            print(f"图片 {filename}: 预测值 = {pred_digit}, 真实值 = {true_label}, "
                 f"置信度 = {confidence:.4f}, 是否正确 = {correct}")
    
    # 计算准确率（如果有真实标签）
    if all(r['correct'] is not None for r in results):
        accuracy = sum(1 for r in results if r['correct']) / len(results)
        print(f"\n总体准确率: {accuracy:.2%}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='手写数字识别预测')
    parser.add_argument('--model', type=str, default='models/finetuned_model.pth', help='模型文件路径')
    parser.add_argument('--image', type=str, help='要预测的单张图片路径')
    parser.add_argument('--dir', type=str, help='要批量预测的图片文件夹路径')
    parser.add_argument('--visualize', action='store_true', help='是否可视化预测结果')
    
    args = parser.parse_args()
    
    # 加载模型
    print(f"加载模型: {args.model}")
    model, _ = load_model(args.model, device)
    
    if args.image:
        # 单张图片预测
        print(f"预测图片: {args.image}")
        predicted, confidence, probabilities = predict_digit(model, args.image)
        print(f"预测结果: {predicted}, 置信度: {confidence:.4f}")
        
        if args.visualize:
            visualize_prediction(args.image, predicted, probabilities)
    
    elif args.dir:
        # 批量预测
        print(f"批量预测文件夹: {args.dir}")
        batch_predict(model, args.dir)
    
    else:
        print("请提供--image或--dir参数")

if __name__ == "__main__":
    main() 