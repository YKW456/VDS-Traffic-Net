#!/usr/bin/env python3
"""
ResNet34模型图片预测脚本 - 46分类干扰光识别
"""
import os
import time
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torchvision import transforms
from train_vds_no_ewc import create_resnet34_pretrained
import glob

class InterferenceLightPredictor:
    def __init__(self, model_path, device=None):
        """
        初始化干扰光类型预测器 - ResNet34模型

        Args:
            model_path: 模型权重文件路径
            device: 计算设备
        """
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 数据预处理（与训练时保持一致）
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # 46个干扰光类别 - 按数字顺序排列，确保索引i对应类别i
        self.class_names = [str(i) for i in range(46)]  # ['0', '1', '2', ..., '45']
        self.num_classes = 46

        # 验证类别映射：索引i应该对应类别名称str(i)
        print(f"📋 类别映射验证:")
        for i in range(min(10, len(self.class_names))):
            print(f"   索引{i} → 类别'{self.class_names[i]}'")
        if len(self.class_names) > 10:
            print(f"   ... (共{len(self.class_names)}个类别)")
        print(f"   索引{len(self.class_names)-1} → 类别'{self.class_names[-1]}'")
        print(f"   ✅ 映射规则: 索引i → 类别'{i}'")

        # 加载模型
        self.model = self._load_model(model_path)

        print(f"✅ ResNet34干扰光预测器初始化完成")
        print(f"   设备: {self.device}")
        print(f"   类别数: {self.num_classes}")
        print(f"   模型架构: ResNet34预训练")
        print(f"   模型路径: {model_path}")
    
    def _load_model(self, model_path):
        """加载训练好的ResNet34模型"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        print(f"🔄 正在加载ResNet34模型: {model_path}")

        # 创建ResNet34模型（46个类别）
        model = create_resnet34_pretrained(num_classes=self.num_classes)

        # 加载权重
        try:
            state_dict = torch.load(model_path, map_location=self.device)

            # 验证权重形状
            if 'fc.weight' in state_dict:
                fc_weight_shape = state_dict['fc.weight'].shape
                model_classes = fc_weight_shape[0]

                if model_classes != self.num_classes:
                    raise ValueError(f"模型类别数不匹配: 期望{self.num_classes}, 实际{model_classes}")

                print(f"✅ 权重验证通过: {model_classes}个类别")

            model.load_state_dict(state_dict)
            print(f"✅ ResNet34模型权重加载成功")

        except Exception as e:
            print(f"❌ ResNet34模型加载失败: {str(e)}")
            raise

        # 设置为评估模式
        model.eval()
        model.to(self.device)

        return model
    
    def predict_single_image(self, image_path):
        """
        预测单张图片
        
        Args:
            image_path: 图片路径
            
        Returns:
            tuple: (predicted_class, class_name, confidence, prediction_time, top5_predictions)
        """
        # 记录开始时间
        start_time = time.time()

        try:
            # 加载和预处理图片
            image = Image.open(image_path).convert('RGB')
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # 模型预测
            with torch.no_grad():
                output = self.model(input_tensor)
                probabilities = torch.softmax(output, dim=1)

                # 获取最高置信度的预测
                confidence, predicted_class = torch.max(probabilities, 1)
                predicted_class = predicted_class.item()
                confidence = confidence.item()
                class_name = self.class_names[predicted_class]

                # 获取Top5预测
                top5_probs, top5_classes = torch.topk(probabilities, 5, dim=1)
                top5_predictions = []
                for i in range(5):
                    cls_idx = top5_classes[0][i].item()
                    cls_prob = top5_probs[0][i].item()
                    top5_predictions.append((cls_idx, self.class_names[cls_idx], cls_prob))

            # 记录结束时间
            end_time = time.time()
            prediction_time = end_time - start_time

            return predicted_class, class_name, confidence, prediction_time, top5_predictions

        except Exception as e:
            print(f"❌ 预测图片失败: {image_path}, 错误: {str(e)}")
            end_time = time.time()
            prediction_time = end_time - start_time
            return -1, "error", 0.0, prediction_time, []
    
    def predict_folder(self, folder_path, output_csv_path):
        """
        预测文件夹中的所有图片
        
        Args:
            folder_path: 图片文件夹路径
            output_csv_path: 输出CSV文件路径
        """
        if not os.path.exists(folder_path):
            print(f"❌ 文件夹不存在: {folder_path}")
            return None
        
        print(f"\n📁 开始预测文件夹: {folder_path}")
        
        # 获取所有图片文件
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(folder_path, ext)))
            image_files.extend(glob.glob(os.path.join(folder_path, ext.upper())))
        
        image_files.sort()  # 按文件名排序
        
        if not image_files:
            print(f"❌ 文件夹中没有找到图片文件: {folder_path}")
            return None
        
        print(f"📊 找到 {len(image_files)} 张图片")
        
        # 预测结果列表
        results = []
        total_time = 0
        
        # 逐张预测
        for i, image_path in enumerate(image_files):
            image_name = os.path.basename(image_path)
            
            # 预测单张图片
            predicted_class, class_name, confidence, prediction_time, top5_predictions = self.predict_single_image(image_path)
            total_time += prediction_time

            # 记录结果
            result = {
                'Image_Name': image_name,
                'Image_Path': image_path,
                'Predicted_Class': predicted_class,
                'Predicted_Class_Name': class_name,
                'Confidence': confidence,
                'Prediction_Time_Seconds': prediction_time,
                'Prediction_Time_MS': prediction_time * 1000,
                'Top2_Class': top5_predictions[1][0] if len(top5_predictions) > 1 else -1,
                'Top2_Confidence': top5_predictions[1][2] if len(top5_predictions) > 1 else 0.0,
                'Top3_Class': top5_predictions[2][0] if len(top5_predictions) > 2 else -1,
                'Top3_Confidence': top5_predictions[2][2] if len(top5_predictions) > 2 else 0.0
            }
            results.append(result)

            # 显示进度
            if (i + 1) % 10 == 0 or i == 0 or i == len(image_files) - 1:
                avg_time = total_time / (i + 1)
                top3_info = f"Top3: {top5_predictions[0][0]},{top5_predictions[1][0] if len(top5_predictions)>1 else 'N/A'},{top5_predictions[2][0] if len(top5_predictions)>2 else 'N/A'}" if top5_predictions else ""
                print(f"   进度: {i+1}/{len(image_files)}, "
                      f"图片: {image_name}, "
                      f"预测: 类别{predicted_class} (置信度: {confidence:.4f}), "
                      f"耗时: {prediction_time*1000:.2f}ms, "
                      f"{top3_info}")
        
        # 保存结果到CSV
        df = pd.DataFrame(results)
        df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
        
        # 统计信息
        avg_time = total_time / len(image_files)
        max_time = max([r['Prediction_Time_Seconds'] for r in results])
        min_time = min([r['Prediction_Time_Seconds'] for r in results])
        
        print(f"\n📊 预测完成统计:")
        print(f"   总图片数: {len(image_files)}")
        print(f"   总耗时: {total_time:.4f}秒")
        print(f"   平均耗时: {avg_time*1000:.2f}ms")
        print(f"   最大耗时: {max_time*1000:.2f}ms")
        print(f"   最小耗时: {min_time*1000:.2f}ms")
        print(f"   结果保存: {output_csv_path}")
        
        # 类别统计
        class_counts = df['Predicted_Class_Name'].value_counts()
        print(f"\n📈 预测类别统计:")
        for class_name, count in class_counts.items():
            percentage = count / len(df) * 100
            print(f"   {class_name}: {count} 张 ({percentage:.1f}%)")
        
        return results


def main():
    """主函数"""
    print("🔮 ResNet34干扰光类型预测与时间统计")
    print("=" * 60)

    # 配置参数
    model_path = "./model_save/train_vds_no_ewc_best_model.pth"  # ResNet34模型路径
    folder_5nd_1 = "5nd_2"  # 第5个功能生成的图片
    folder_6nd_1 = "6nd_2"  # 第6个功能生成的图片
    output_dir = "data_save"

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        print("请先运行 train_vds_no_ewc.py 训练ResNet34模型")
        return
    
    # 检查文件夹是否存在
    folders_to_predict = []
    if os.path.exists(folder_5nd_1):
        folders_to_predict.append((folder_5nd_1, "5nd_1_resnet34_predictions.csv"))
    else:
        print(f"⚠️ 文件夹不存在: {folder_5nd_1}")

    if os.path.exists(folder_6nd_1):
        folders_to_predict.append((folder_6nd_1, "6nd_1_resnet34_predictions.csv"))
    else:
        print(f"⚠️ 文件夹不存在: {folder_6nd_1}")

    if not folders_to_predict:
        print("❌ 没有找到要预测的文件夹")
        return

    try:
        # 初始化预测器
        predictor = InterferenceLightPredictor(model_path)
        
        # 记录总开始时间
        total_start_time = time.time()
        
        # 对每个文件夹进行预测
        all_results = {}
        for folder_path, output_filename in folders_to_predict:
            output_csv_path = os.path.join(output_dir, output_filename)
            results = predictor.predict_folder(folder_path, output_csv_path)
            if results:
                all_results[folder_path] = results
        
        # 记录总结束时间
        total_end_time = time.time()
        total_duration = total_end_time - total_start_time
        
        # 总体统计
        print(f"\n" + "=" * 60)
        print(f"🎉 所有预测完成！")
        print(f"📊 总体统计:")
        print(f"   总耗时: {total_duration:.4f}秒")
        print(f"   预测文件夹数: {len(all_results)}")
        
        total_images = sum(len(results) for results in all_results.values())
        if total_images > 0:
            avg_time_per_image = total_duration / total_images
            print(f"   总图片数: {total_images}")
            print(f"   平均每张图片耗时: {avg_time_per_image*1000:.2f}ms")
        
        print(f"\n📁 输出文件:")
        for folder_path, output_filename in folders_to_predict:
            if folder_path in all_results:
                output_path = os.path.join(output_dir, output_filename)
                print(f"   {folder_path} -> {output_path}")
        
    except Exception as e:
        print(f"❌ 预测过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
