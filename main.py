import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import json
from datetime import datetime
from model import DigitRecognitionCNN, CustomHandwrittenDataset

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 检查设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'使用设备: {device}')

# 数据预处理
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

# 加载MNIST数据集
def load_mnist_data(batch_size=64):
    train_dataset = datasets.MNIST(root='./data', train=True, 
                                 transform=transform_train, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, 
                                transform=transform_test, download=True)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                            shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                           shuffle=False, num_workers=0)
    
    return train_loader, test_loader

# 训练函数
def train_model(model, train_loader, val_loader, epochs, lr=0.001, device='cpu'):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
    
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch+1}/{epochs} [{batch_idx*len(data)}/{len(train_loader.dataset)} '
                      f'({100.*batch_idx/len(train_loader):.0f}%)] Loss: {loss.item():.6f}')
        
        # 计算训练集准确率
        train_acc = 100. * correct / total
        train_losses.append(train_loss / len(train_loader))
        train_accs.append(train_acc)
        
        # 验证阶段
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        val_acc = 100. * correct / total
        val_losses.append(val_loss / len(val_loader))
        val_accs.append(val_acc)
        
        scheduler.step(val_loss)
        
        print(f'Epoch: {epoch+1}/{epochs} - Train Loss: {train_losses[-1]:.4f}, '
              f'Train Acc: {train_acc:.2f}%, Val Loss: {val_losses[-1]:.4f}, Val Acc: {val_acc:.2f}%')
    
    return train_losses, val_losses, train_accs, val_accs

# 微调函数（用于自定义数据）
def finetune_model(model, custom_train_loader, custom_val_loader, epochs=10, lr=0.0001, device='cpu'):
    # 冻结前面的层，只训练最后几层
    for param in model.parameters():
        param.requires_grad = False
    
    # 解冻最后的全连接层
    for param in model.fc2.parameters():
        param.requires_grad = True
    for param in model.fc3.parameters():
        param.requires_grad = True
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    
    print("开始微调模型...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(custom_train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
        
        train_acc = 100. * correct / total
        print(f'微调 Epoch: {epoch+1}/{epochs} - Loss: {train_loss/len(custom_train_loader):.4f}, '
              f'Acc: {train_acc:.2f}%')
        
        # 评估在验证集上的性能
        if custom_val_loader:
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for data, target in custom_val_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    val_loss += criterion(output, target).item()
                    _, predicted = output.max(1)
                    val_total += target.size(0)
                    val_correct += predicted.eq(target).sum().item()
            
            val_acc = 100. * val_correct / val_total
            print(f'微调 验证 Epoch: {epoch+1}/{epochs} - Loss: {val_loss/len(custom_val_loader):.4f}, '
                  f'Acc: {val_acc:.2f}%')
    
    return model

# 评估函数
def evaluate_model(model, test_loader, device='cpu'):
    model.eval()
    correct = 0
    total = 0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            c = (predicted == target).squeeze()
            for i in range(target.size(0)):
                label = target[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    
    print(f'总体准确率: {100 * correct / total:.2f}%')
    
    for i in range(10):
        if class_total[i] > 0:
            print(f'数字 {i} 的准确率: {100 * class_correct[i]/class_total[i]:.2f}%')
    
    return 100 * correct / total

# 可视化函数
def plot_training_history(train_losses, val_losses, train_accs, val_accs):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # 损失曲线
    ax1.plot(train_losses, label='训练损失')
    ax1.plot(val_losses, label='验证损失')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('训练和验证损失')
    ax1.legend()
    ax1.grid(True)
    
    # 准确率曲线
    ax2.plot(train_accs, label='训练准确率')
    ax2.plot(val_accs, label='验证准确率')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('训练和验证准确率')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

# 保存和加载模型
def save_model(model, optimizer, epoch, loss, path='model_checkpoint.pth'):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'model_config': {
            'num_classes': 10,
            'architecture': 'DigitRecognitionCNN'
        }
    }, path)
    print(f'模型已保存到 {path}')

def create_sample_dataset():
    """创建用于微调的手写数据集文件夹结构"""
    os.makedirs('custom_digits/train', exist_ok=True)
    os.makedirs('custom_digits/val', exist_ok=True)
    
    print("请将您的手写学号图片按以下格式命名并放入对应文件夹：")
    print("- 训练数据: custom_digits/train/")
    print("- 验证数据: custom_digits/val/")
    print("- 文件命名格式: {数字}_{序号}.png (例如: 0_001.png, 1_002.png)")
    
# 主程序
if __name__ == "__main__":
    # 设置参数
    batch_size = 64
    epochs_pretrain = 10
    epochs_finetune = 5
    learning_rate = 0.001
    
    # 1. 加载MNIST数据集
    print("加载MNIST数据集...")
    train_loader, test_loader = load_mnist_data(batch_size)
    
    # 2. 创建模型
    from model import DigitRecognitionCNN
    model = DigitRecognitionCNN(num_classes=10).to(device)
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")
    
    # 3. 在MNIST上预训练
    print("\n开始在MNIST数据集上预训练...")
    train_losses, val_losses, train_accs, val_accs = train_model(
        model, train_loader, test_loader, epochs_pretrain, learning_rate, device
    )
    
    # 4. 可视化训练过程
    plot_training_history(train_losses, val_losses, train_accs, val_accs)
    
    # 5. 评估预训练模型
    print("\n评估预训练模型在MNIST测试集上的性能...")
    evaluate_model(model, test_loader, device)
    
    # 6. 保存预训练模型
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    save_model(model, optimizer, epochs_pretrain, train_losses[-1], 'models/mnist_pretrained_model.pth')
    
    # 7. 准备自定义数据集
    create_sample_dataset()
    
    # 检查是否存在自定义数据
    if os.path.exists('custom_digits/train') and len(os.listdir('custom_digits/train')) > 0:
        print("\n加载自定义手写学号数据集...")
        custom_train_dataset = CustomHandwrittenDataset('custom_digits/train', transform_train)
        custom_val_dataset = CustomHandwrittenDataset('custom_digits/val', transform_test)
        
        custom_train_loader = DataLoader(custom_train_dataset, batch_size=16, shuffle=True)
        custom_val_loader = DataLoader(custom_val_dataset, batch_size=16, shuffle=False)
        
        # 8. 微调模型
        model = finetune_model(model, custom_train_loader, custom_val_loader, 
                      epochs_finetune, learning_rate*0.1, device)
        
        # 9. 保存微调后的模型
        save_model(model, optimizer, epochs_finetune, 0, 'models/finetuned_model.pth')
        
        # 10. 评估微调后的模型
        if len(custom_val_dataset) > 0:
            print("\n评估微调后的模型在自定义验证集上的性能...")
            evaluate_model(model, custom_val_loader, device)
    else:
        print("\n未找到自定义数据集，跳过微调步骤。")
        print("请按照上述说明准备您的手写学号数据。")
    
    print("\n训练完成！")