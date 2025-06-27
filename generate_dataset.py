import os
import argparse
from data_utils import generate_custom_dataset, augment_dataset, visualize_dataset

def main():
    parser = argparse.ArgumentParser(description='生成手写学号数据集')
    parser.add_argument('--output_dir', type=str, default='./custom_digits', help='数据集输出目录')
    parser.add_argument('--samples', type=int, default=30, help='每个数字的样本数量')
    parser.add_argument('--augment', action='store_true', help='是否进行数据增强')
    parser.add_argument('--augment_factor', type=int, default=3, help='数据增强倍数')
    parser.add_argument('--visualize', action='store_true', help='是否可视化数据集')
    
    args = parser.parse_args()
    
    print("开始生成基础数据集...")
    generate_custom_dataset(args.output_dir, num_samples_per_digit=args.samples)
    
    if args.augment:
        print("\n开始进行数据增强...")
        augmented_train_dir = os.path.join(args.output_dir, 'train_augmented')
        augment_dataset(os.path.join(args.output_dir, 'train'), 
                       augmented_train_dir, 
                       augmentation_factor=args.augment_factor)
        
        if args.visualize:
            print("\n可视化增强后的数据集...")
            visualize_dataset(augmented_train_dir, num_samples=15)
    elif args.visualize:
        print("\n可视化原始数据集...")
        visualize_dataset(os.path.join(args.output_dir, 'train'), num_samples=15)
    
    print("\n数据集准备完成!")
    print(f"训练集位置: {os.path.join(args.output_dir, 'train' if not args.augment else 'train_augmented')}")
    print(f"验证集位置: {os.path.join(args.output_dir, 'val')}")
    
    # 打印使用说明
    print("\n使用说明:")
    print("1. 使用如下命令预训练模型:")
    print("   python main.py")
    print("2. 如果您已经有自己的手写学号图片，请将图片按以下格式命名并放到相应文件夹:")
    print("   - 训练图片: custom_digits/train/")
    print("   - 验证图片: custom_digits/val/")
    print("   - 文件命名格式: {数字}_{序号}.png (例如: 0_001.png, 1_002.png)")
    print("3. 然后运行以下命令进行微调:")
    print("   python main.py --finetune")
    
if __name__ == "__main__":
    main() 