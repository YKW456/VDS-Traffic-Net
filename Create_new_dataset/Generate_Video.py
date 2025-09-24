import os
import cv2
import numpy as np
import warnings
import shutil
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial

# Fix np.bool issue
if not hasattr(np, 'bool'):
    np.bool = bool
    warnings.warn("Patched np.bool to bool for imgaug compatibility", RuntimeWarning)
from imgaug import augmenters as iaa
from generate_jiguang_model import generate_moving_laser_sequence, generate_moving_deformation_sequence, generate_3d_perspective_sequence, generate_4th_sequence_with_interference, generate_5th_sequence_with_moving_lights, generate_6th_x_axis_rotation_sequence


def generate_moving_laser_video():
    """
    First function: Generate moving light spot sequence
    - Generate 60 new images from test2.jpg
    - 12 images per group, 5 groups total (5 types of interference light)
    - Fixed interference light intensity within each group, continuous position changes
    - No spatial distortion effects
    """
    print("=" * 60)
    print("Starting to generate moving light spot sequence...")
    print("=" * 60)

    # Check if input file exists
    input_image = "test2.jpg"
    if not os.path.exists(input_image):
        print(f"❌ Error: Cannot find input image {input_image}")
        print("Please ensure test2.jpg file exists in current directory")
        return

    # Output directory
    output_dir = "moving_laser_dataset"

    # Parameter settings
    glow_size_range = (1835, 2225)  # Glow size range
    center_brightness = 255         # Center brightness

    try:
        # Call the first function
        generate_moving_laser_sequence(
            image_path=input_image,
            output_dir=output_dir,
            glow_size_range=glow_size_range,
            center_brightness=center_brightness
        )

        print("✅ Moving light spot sequence generation completed!")
        print(f"📁 Output directory: {output_dir}")
        print("📊 Generation statistics:")
        print("   - Total images: 60")
        print("   - Grouping method: 5 interference lights × 12 images/group")
        print("   - Effect characteristics: Continuous light spot position movement, no spatial deformation")

    except Exception as e:
        print(f"❌ Error generating moving light spot sequence: {str(e)}")


def generate_deformation_video():
    """
    Second function: Generate spatial deformation sequence
    - Generate 60 new images from test2.jpg
    - 12 images per group, 5 groups total (5 types of interference light)
    - Each group randomly selects one deformation method (perspective/shear/scale)
    - Spatial distortion effects change continuously, simulating vehicle movement
    - Fixed interference light position
    """
    print("=" * 60)
    print("Starting to generate spatial deformation sequence...")
    print("=" * 60)

    # Check if input file exists
    input_image = "test2.jpg"
    if not os.path.exists(input_image):
        print(f"❌ Error: Cannot find input image {input_image}")
        print("Please ensure test2.jpg file exists in current directory")
        return

    # Output directory
    output_dir = "deformation_dataset"

    # Parameter settings
    glow_size_range = (1835, 2225)  # Glow size range
    center_brightness = 255         # Center brightness

    try:
        # Call the second function
        generate_moving_deformation_sequence(
            image_path=input_image,
            output_dir=output_dir,
            glow_size_range=glow_size_range,
            center_brightness=center_brightness
        )

        print("✅ Spatial deformation sequence generation completed!")
        print(f"📁 Output directory: {output_dir}")
        print("📊 Generation statistics:")
        print("   - Total images: 60")
        print("   - Grouping method: 5 interference lights × 12 images/group")
        print("   - Deformation types: Perspective transform/Shear transform/Scale transform")
        print("   - Effect characteristics: Continuous spatial deformation changes, fixed light spot position")

    except Exception as e:
        print(f"❌ Error generating spatial deformation sequence: {str(e)}")


def generate_3d_perspective_video():
    """
    Third function: Generate pure 3D perspective transformation sequence (no interference light)
    - Generate 60 continuous transformation images from test2.jpg
    - Simulate complete process of vehicle approaching from distance
    - Three-dimensional spatial sense from front view to right side view
    - No interference light, focus on spatial transformation effects
    """
    print("=" * 60)
    print("Starting to generate pure 3D perspective transformation sequence...")
    print("=" * 60)

    # Check if input file exists
    input_image = "test2.jpg"
    if not os.path.exists(input_image):
        print(f"❌ Error: Cannot find input image {input_image}")
        print("Please ensure test2.jpg file exists in current directory")
        return

    # Output directory
    output_dir = "3rd_1"

    try:
        # Call the third function (no light spot parameters needed)
        generate_3d_perspective_sequence(
            image_path=input_image,
            output_dir=output_dir,
            glow_size_range=None,
            center_brightness=None
        )

        print("✅ Pure 3D perspective transformation sequence generation completed!")
        print(f"📁 Output directory: {output_dir}")
        print("📊 Generation statistics:")
        print("   - Total images: 60")
        print("   - Transformation method: Continuous 3D perspective transformation")
        print("   - Distance effect: From far to near (0.7x to 1.2x scaling)")
        print("   - Viewing angle change: From front view (0°) to right side view (45°)")
        print("   - Height change: Slight vertical tilt (0° to 8°)")
        print("   - Lighting effect: Simulate lighting changes when vehicle approaches")
        print("   - Effect characteristics: Three-dimensional spatial sense, no interference light")

    except Exception as e:
        print(f"❌ Error generating 3D perspective transformation sequence: {str(e)}")


def generate_4th_interference_video():
    """
    Fourth function: Add Create_dataset style interference light to 3rd_1 images
    - Based on 60 3D transformation images generated by the third function
    - Add white circular interference light, consistent with training data generation method
    - Interference light size continuously changes from 20% to 80% of image area
    - Interference light position changes continuously with moderate variation range
    """
    print("=" * 60)
    print("Starting to add Create_dataset style interference light to 3rd_1 images...")
    print("=" * 60)

    # Check if source directory exists
    source_dir = "3rd_1"
    if not os.path.exists(source_dir):
        print(f"❌ Error: Cannot find source directory {source_dir}")
        print("Please run the third function first to generate 3rd_1 directory")
        return

    # Output directory
    output_dir = "4nd_1"

    try:
        # Call the fourth function
        generate_4th_sequence_with_interference(
            source_dir=source_dir,
            output_dir=output_dir
        )

        print("✅ Fourth function: Add interference light sequence generation completed!")
        print(f"📁 Source directory: {source_dir}")
        print(f"📁 Output directory: {output_dir}")
        print("📊 Generation statistics:")
        print("   - Total images: 60")
        print("   - Base images: 3D perspective transformation sequence")
        print("   - Interference light color: White")
        print("   - Interference light shape: Circular")
        print("   - Interference light transparency: 12 images per group [0.7,0.78,0.85,0.92,1.0]")
        print("   - Size variation: From 20% to 80% of image area")
        print("   - Position variation: Continuous movement with moderate variation range")
        print("   - Generation method: Completely consistent with Create_dataset.py")

    except Exception as e:
        print(f"❌ Error generating fourth function: {str(e)}")


def generate_5th_moving_lights_video():
    """
    Fifth function: Generate fixed-size, continuously moving interference light sequence
    - Based on original image (no distortion effects)
    - Fixed interference light size at 25% of image area
    - 3 colors: Red, Green, White
    - 3 transparency levels: 1.0, 0.92, 0.85
    - 3 movement modes: Left→Right, Right→Left, Top→Bottom
    """
    print("=" * 60)
    print("Starting to generate fixed-size, continuously moving interference light sequence...")
    print("=" * 60)

    # Check if input file exists
    input_image = "test2.jpg"
    if not os.path.exists(input_image):
        print(f"❌ Error: Cannot find input image {input_image}")
        print("Please ensure test2.jpg file exists in current directory")
        return

    # Output directory
    output_dir = "5nd_2"

    try:
        # Call the fifth function
        generate_5th_sequence_with_moving_lights(
            image_path=input_image,
            output_dir=output_dir
        )

        print("✅ Fifth function: Fixed-size moving interference light sequence generation completed!")
        print(f"📁 Output directory: {output_dir}")
        print("📊 Generation statistics:")
        print("   - Total images: 60")
        print("   - Base images: Original image (no distortion)")
        print("   - Interference light size: Fixed 25% area")
        print("   - Interference light colors: Red, Green, White (3 types)")
        print("   - Interference light transparency: 1.0, 0.92, 0.85 (3 types)")
        print("   - Movement modes: First 20 Left→Right, Middle 20 Right→Left, Last 20 Top→Bottom")
        print("   - Position changes: Continuous movement within each group")

    except Exception as e:
        print(f"❌ Error generating fifth function: {str(e)}")


def generate_6th_x_rotation_video():
    """
    Sixth function: Generate image sequence rotating around X-axis
    - Based on test6.jpg original image
    - X-axis rotation from 0° to 75°
    - Simulate from front view to top-down view angle
    - 60 continuous rotation images
    """
    print("=" * 60)
    print("Starting to generate image sequence rotating around X-axis...")
    print("=" * 60)


    # input_image = "test6.jpg"
    # if not os.path.exists(input_image):
    #     print(f"❌ Error: Cannot find input image {input_image}")
    #     print("Please ensure test6.jpg file exists in current directory")
    #     return
    #
    # # Output directory
    # output_dir = "6nd_1"
    # Check if input file exists
    name = '225'
    input_image = "test"+name+".jpg"
    if not os.path.exists(input_image):
        print(f"❌ Error: Cannot find input image {input_image}")
        print("Please ensure test6.jpg file exists in current directory")
        return

    # Output directory
    output_dir = "6nd_"+name


    try:
        # 调用第六个函数
        generate_6th_x_axis_rotation_sequence(
            image_path=input_image,
            output_dir=output_dir
        )

        print("✅ 第六个功能：绕X轴旋转序列生成完成！")
        print(f"📁 输出目录: {output_dir}")
        print("📊 生成统计:")
        print("   - 总图片数: 60张")
        print("   - 基础图片: test6.jpg")
        print("   - 旋转轴: X轴")
        print("   - 旋转角度: 0°到75°")
        print("   - 旋转效果: 从正面视角到俯视角度")
        print("   - 透视变换: 三维旋转透视效果")

    except Exception as e:
        print(f"❌ 生成第六个功能时出错: {str(e)}")





def main():
    """主函数 - 交互式菜单"""
    generate_5th_moving_lights_video()
    # generate_6th_x_rotation_video()



if __name__ == "__main__":
    main()


