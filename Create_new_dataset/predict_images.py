#!/usr/bin/env python3
"""
ResNet34 image prediction script for 46-class interference light classification
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
        Initialize interference light classifier based on ResNet34

        Args:
            model_path: file path of trained model checkpoint
            device: computation device (cuda/cpu)
        """
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Image preprocessing pipeline (consistent with training transforms)
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # Define 46 interference light classes, index i maps to class name str(i)
        self.class_names = [str(i) for i in range(46)]  # ['0', '1', '2', ..., '45']
        self.num_classes = 46

        # Verify index-to-classname mapping
        print(f"📋 Class Mapping Check:")
        for i in range(min(10, len(self.class_names))):
            print(f"   Index {i} → Class '{self.class_names[i]}'")
        if len(self.class_names) > 10:
            print(f"   ... (total {len(self.class_names)} classes)")
        print(f"   Index {len(self.class_names)-1} → Class '{self.class_names[-1]}'")
        print(f"   ✅ Mapping rule: Index i → Class name '{i}'")

        # Load trained ResNet34 model
        self.model = self._load_model(model_path)

        print(f"✅ ResNet34 Interference Light Predictor initialized")
        print(f"   Device: {self.device}")
        print(f"   Total classes: {self.num_classes}")
        print(f"   Backbone: Pretrained ResNet34")
        print(f"   Checkpoint path: {model_path}")
    
    def _load_model(self, model_path):
        """Load pre-trained ResNet34 checkpoint from local file"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Checkpoint file not found: {model_path}")

        print(f"🔄 Loading ResNet34 checkpoint: {model_path}")

        # Build ResNet34 architecture configured for 46 output classes
        model = create_resnet34_pretrained(num_classes=self.num_classes)

        # Load saved state dict
        try:
            state_dict = torch.load(model_path, map_location=self.device)

            # Validate FC layer output dimension matches target class count
            if 'fc.weight' in state_dict:
                fc_weight_shape = state_dict['fc.weight'].shape
                model_classes = fc_weight_shape[0]

                if model_classes != self.num_classes:
                    raise ValueError(f"Class number mismatch: expected {self.num_classes}, checkpoint has {model_classes}")

                print(f"✅ Weight dimension verified: {model_classes} output classes")

            model.load_state_dict(state_dict)
            print(f"✅ ResNet34 weights loaded successfully")

        except Exception as e:
            print(f"❌ Failed to load ResNet34 model: {str(e)}")
            raise

        # Switch model to evaluation mode
        model.eval()
        model.to(self.device)

        return model
    
    def predict_single_image(self, image_path):
        """
        Run inference on single input image
        
        Args:
            image_path: absolute/relative path of input image
            
        Returns:
            tuple: (predicted_class_id, class_name, confidence, inference_time, top5_pred_list)
        """
        # Record inference start timestamp
        start_time = time.time()

        try:
            # Open RGB image and apply preprocessing
            image = Image.open(image_path).convert('RGB')
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Forward pass without gradient computation
            with torch.no_grad():
                output = self.model(input_tensor)
                probabilities = torch.softmax(output, dim=1)

                # Fetch top-1 prediction and confidence score
                confidence, predicted_class = torch.max(probabilities, 1)
                predicted_class = predicted_class.item()
                confidence = confidence.item()
                class_name = self.class_names[predicted_class]

                # Extract top5 candidate predictions
                top5_probs, top5_classes = torch.topk(probabilities, 5, dim=1)
                top5_predictions = []
                for i in range(5):
                    cls_idx = top5_classes[0][i].item()
                    cls_prob = top5_probs[0][i].item()
                    top5_predictions.append((cls_idx, self.class_names[cls_idx], cls_prob))

            # Calculate single image inference cost
            end_time = time.time()
            prediction_time = end_time - start_time

            return predicted_class, class_name, confidence, prediction_time, top5_predictions

        except Exception as e:
            print(f"❌ Prediction failed for {image_path}, error: {str(e)}")
            end_time = time.time()
            prediction_time = end_time - start_time
            return -1, "error", 0.0, prediction_time, []
    
    def predict_folder(self, folder_path, output_csv_path):
        """
        Batch predict all images under specified directory
        
        Args:
            folder_path: target image folder path
            output_csv_path: full path to save prediction csv result
        """
        if not os.path.exists(folder_path):
            print(f"❌ Target folder not found: {folder_path}")
            return None
        
        print(f"\n📁 Starting batch prediction on folder: {folder_path}")
        
        # Collect all valid image files with supported extensions
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(folder_path, ext)))
            image_files.extend(glob.glob(os.path.join(folder_path, ext.upper())))
        
        image_files.sort()  # Sort image list by filename
        
        if not image_files:
            print(f"❌ No valid image found inside {folder_path}")
            return None
        
        print(f"📊 Total detected images: {len(image_files)}")
        
        results = []
        total_time = 0
        
        # Iterate and predict each image one by one
        for i, image_path in enumerate(image_files):
            image_name = os.path.basename(image_path)
            
            predicted_class, class_name, confidence, prediction_time, top5_predictions = self.predict_single_image(image_path)
            total_time += prediction_time

            # Append prediction metadata to result dict
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

            # Print progress every 10 images or at first/last sample
            if (i + 1) % 10 == 0 or i == 0 or i == len(image_files) - 1:
                avg_time = total_time / (i + 1)
                top3_info = f"Top3: {top5_predictions[0][0]},{top5_predictions[1][0] if len(top5_predictions)>1 else 'N/A'},{top5_predictions[2][0] if len(top5_predictions)>2 else 'N/A'}" if top5_predictions else ""
                print(f"   Progress: {i+1}/{len(image_files)}, "
                      f"File: {image_name}, "
                      f"Pred: Class {predicted_class} (Conf: {confidence:.4f}), "
                      f"Latency: {prediction_time*1000:.2f}ms, "
                      f"{top3_info}")
        
        # Export all results into CSV file
        df = pd.DataFrame(results)
        df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
        
        # Compute overall inference statistics
        avg_time = total_time / len(image_files)
        max_time = max([r['Prediction_Time_Seconds'] for r in results])
        min_time = min([r['Prediction_Time_Seconds'] for r in results])
        
        print(f"\n📊 Batch Prediction Summary:")
        print(f"   Total images: {len(image_files)}")
        print(f"   Total elapsed time: {total_time:.4f}s")
        print(f"   Avg single image latency: {avg_time*1000:.2f}ms")
        print(f"   Max single image latency: {max_time*1000:.2f}ms")
        print(f"   Min single image latency: {min_time*1000:.2f}ms")
        print(f"   Saved result to: {output_csv_path}")
        
        # Count prediction distribution per class
        class_counts = df['Predicted_Class_Name'].value_counts()
        print(f"\n📈 Class prediction distribution:")
        for class_name, count in class_counts.items():
            percentage = count / len(df) * 100
            print(f"   Class {class_name}: {count} samples ({percentage:.1f}%)")
        
        return results


def main():
    """Main program entry"""
    print("🔮 ResNet34 Interference Light Prediction & Latency Statistics")
    print("=" * 60)

    # Runtime configuration parameters
    model_path = "./model_save/train_vds_no_ewc_best_model.pth"  # ResNet34 checkpoint path
    folder_5nd_1 = "5nd_2"  # Image folder generated from No.5 module
    folder_6nd_1 = "6nd_2"  # Image folder generated from No.6 module
    output_dir = "data_save"

    # Create output directory if not exists
    os.makedirs(output_dir, exist_ok=True)

    # Check checkpoint existence
    if not os.path.exists(model_path):
        print(f"❌ Model checkpoint missing: {model_path}")
        print("Please run train_vds_no_ewc.py first to train ResNet34 model")
        return
    
    # Collect valid target folders for prediction
    folders_to_predict = []
    if os.path.exists(folder_5nd_1):
        folders_to_predict.append((folder_5nd_1, "5nd_1_resnet34_predictions.csv"))
    else:
        print(f"⚠️ Target folder missing: {folder_5nd_1}")

    if os.path.exists(folder_6nd_1):
        folders_to_predict.append((folder_6nd_1, "6nd_1_resnet34_predictions.csv"))
    else:
        print(f"⚠️ Target folder missing: {folder_6nd_1}")

    if not folders_to_predict:
        print("❌ No valid folders available for prediction")
        return

    try:
        # Initialize prediction instance
        predictor = InterferenceLightPredictor(model_path)
        
        total_start_time = time.time()
        
        all_results = {}
        # Process each target folder sequentially
        for folder_path, output_filename in folders_to_predict:
            output_csv_path = os.path.join(output_dir, output_filename)
            results = predictor.predict_folder(folder_path, output_csv_path)
            if results:
                all_results[folder_path] = results
        
        total_end_time = time.time()
        total_duration = total_end_time - total_start_time
        
        # Global total statistics
        print(f"\n" + "=" * 60)
        print(f"🎉 All prediction tasks finished!")
        print(f"📊 Global Summary:")
        print(f"   Total wall time: {total_duration:.4f}s")
        print(f"   Processed folder count: {len(all_results)}")
        
        total_images = sum(len(results) for results in all_results.values())
        if total_images > 0:
            avg_time_per_image = total_duration / total_images
            print(f"   Total processed images: {total_images}")
            print(f"   Global average latency per image: {avg_time_per_image*1000:.2f}ms")
        
        print(f"\n📁 Generated output files:")
        for folder_path, output_filename in folders_to_predict:
            if folder_path in all_results:
                output_path = os.path.join(output_dir, output_filename)
                print(f"   {folder_path} -> {output_path}")
        
    except Exception as e:
        print(f"❌ Runtime error during prediction: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
