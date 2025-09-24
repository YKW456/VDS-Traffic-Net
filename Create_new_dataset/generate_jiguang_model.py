import cv2
import numpy as np
import random
import os
import shutil

def _make_odd(x):
    x = max(1, int(x))
    return x if x % 2 == 1 else x - 1 if x > 1 else 1

def apply_continuous_deformation(image, strategy, deform_factor):
    """
    应用连续变形，确保变形效果的连续性和可预测性

    Args:
        image: 输入图像
        strategy: 变形策略 ('perspective', 'shear', 'scale')
        deform_factor: 变形强度 (0.0-1.0)

    Returns:
        变形后的图像
    """
    if image is None or deform_factor <= 0:
        return image.copy() if image is not None else None

    height, width = image.shape[:2]

    if strategy == 'perspective':
        return apply_continuous_perspective(image, deform_factor)
    elif strategy == 'shear':
        return apply_continuous_shear(image, deform_factor)
    else:  # scale
        return apply_continuous_scale(image, deform_factor)

def apply_continuous_perspective(image, factor):
    """连续透视变形 - 使用网格变形实现强连续性"""
    height, width = image.shape[:2]

    # 使用网格变形实现更明显的连续扭曲效果
    return apply_grid_distortion(image, factor, 'perspective')

def apply_grid_distortion(image, factor, distortion_type):
    """使用网格变形实现强连续性扭曲"""
    height, width = image.shape[:2]

    # 创建变形映射
    map_x = np.zeros((height, width), dtype=np.float32)
    map_y = np.zeros((height, width), dtype=np.float32)

    # 使用指数增长确保连续性明显
    exponential_factor = factor * factor * factor  # 立方增长

    for y in range(height):
        for x in range(width):
            # 归一化坐标
            norm_x = x / width
            norm_y = y / height

            if distortion_type == 'perspective':
                # 有规律的透视扭曲：模拟车辆接近时的真实变形
                # 使用简单但有效的变形模式，确保规律性和连续性

                # 1. 主要变形：梯形透视扭曲（模拟车辆接近）
                # 上部收缩，下部保持，模拟透视效果
                perspective_strength = exponential_factor * 1.5  # 增强连续性

                # 计算垂直位置的影响权重
                vertical_weight = norm_y  # 0到1，上部影响小，下部影响大

                # 梯形扭曲：上部向中心收缩
                trapezoid_offset = perspective_strength * (1.0 - vertical_weight) * width * 0.3
                new_x = x + (0.5 - norm_x) * trapezoid_offset

                # 2. 辅助变形：垂直压缩（模拟距离变化）
                # 上部压缩更明显，下部压缩较少
                vertical_compression = perspective_strength * 0.8
                compression_factor = 1.0 - vertical_compression * (1.0 - vertical_weight) * 0.4
                new_y = y * compression_factor

                # 3. 微调：轻微的水平拉伸（增强透视感）
                # 中心区域拉伸，边缘区域压缩
                horizontal_stretch = perspective_strength * 0.6
                center_distance = abs(norm_x - 0.5)  # 到中心的距离
                stretch_factor = 1.0 + horizontal_stretch * (0.5 - center_distance) * vertical_weight * 0.2
                new_x = (new_x - width * 0.5) * stretch_factor + width * 0.5

                # 4. 精细调整：边缘柔化（避免突兀变形）
                # 在边缘区域减少变形强度，使过渡更自然
                edge_distance_x = min(norm_x, 1.0 - norm_x)  # 到左右边缘的最小距离
                edge_distance_y = min(norm_y, 1.0 - norm_y)  # 到上下边缘的最小距离
                edge_factor = min(edge_distance_x, edge_distance_y) * 4  # 边缘柔化系数
                edge_factor = min(1.0, edge_factor)  # 限制在0-1之间

                # 应用边缘柔化
                new_x = x + (new_x - x) * edge_factor
                new_y = y + (new_y - y) * edge_factor

            else:
                new_x = x
                new_y = y

            # 确保坐标在合理范围内
            map_x[y, x] = np.clip(new_x, 0, width-1)
            map_y[y, x] = np.clip(new_y, 0, height-1)

    # 应用重映射
    result = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    return result

def apply_barrel_distortion(image, strength):
    """应用桶形畸变增强扭曲效果"""
    height, width = image.shape[:2]

    # 创建畸变映射
    map_x = np.zeros((height, width), dtype=np.float32)
    map_y = np.zeros((height, width), dtype=np.float32)

    center_x, center_y = width // 2, height // 2
    max_radius = min(center_x, center_y)

    for y in range(height):
        for x in range(width):
            # 计算到中心的距离
            dx = x - center_x
            dy = y - center_y
            radius = np.sqrt(dx*dx + dy*dy)

            if radius > 0:
                # 桶形畸变公式
                normalized_radius = radius / max_radius
                distorted_radius = radius * (1 + strength * normalized_radius * normalized_radius)

                # 计算新坐标
                scale = distorted_radius / radius
                new_x = center_x + dx * scale
                new_y = center_y + dy * scale

                map_x[y, x] = new_x
                map_y[y, x] = new_y
            else:
                map_x[y, x] = x
                map_y[y, x] = y

    # 应用重映射
    result = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    return result

def apply_continuous_shear(image, factor):
    """连续剪切变形"""
    height, width = image.shape[:2]

    # 剪切强度与factor成正比
    shear_x = factor * 0.8  # 增强系数
    shear_y = factor * 0.6  # 增强系数

    # 剪切变换矩阵
    shear_matrix = np.float32([
        [1, shear_x, 0],
        [shear_y, 1, 0]
    ])

    # 应用剪切变换
    result = cv2.warpAffine(image, shear_matrix, (width, height),
                           borderMode=cv2.BORDER_CONSTANT,
                           borderValue=(255, 255, 255))
    return result

def apply_continuous_scale(image, factor):
    """连续缩放变形"""
    height, width = image.shape[:2]

    # 缩放强度与factor成正比
    scale_x = 1.0 + factor * 1.5  # 增强系数
    scale_y = 1.0 + factor * 1.2  # 增强系数

    # 缩放变换矩阵（以图像中心为基准）
    center_x, center_y = width // 2, height // 2
    scale_matrix = cv2.getRotationMatrix2D((center_x, center_y), 0, 1.0)
    scale_matrix[0, 0] = scale_x
    scale_matrix[1, 1] = scale_y

    # 应用缩放变换
    result = cv2.warpAffine(image, scale_matrix, (width, height),
                           borderMode=cv2.BORDER_CONSTANT,
                           borderValue=(255, 255, 255))
    return result

def apply_random_stretch(image, stretch_range=(0.0, 0.3)):
    """
    模拟车辆从远处驶向告示牌的效果：智能变形，最小化白色填充

    Args:
        image: 输入图像
        stretch_range: 拉伸/压缩范围，(最小变形比例, 最大变形比例)

    Returns:
        变形后的图像（保持原尺寸，智能填充）
    """
    if image is None:
        return None

    height, width = image.shape[:2]

    # 随机生成变形参数
    stretch_factor = random.uniform(stretch_range[0], stretch_range[1])

    if stretch_factor <= 0:
        return image.copy()

    # 随机选择变形策略
    strategy = random.choice(['perspective', 'shear', 'scale_only'])

    if strategy == 'perspective':
        # 透视变换：模拟车辆接近的透视效果
        return apply_perspective_transform(image, stretch_factor)
    elif strategy == 'shear':
        # 剪切变换：模拟角度变化
        return apply_shear_transform(image, stretch_factor)
    else:
        # 纯缩放：模拟距离变化
        return apply_scale_transform(image, stretch_factor)

def apply_perspective_transform(image, stretch_factor):
    """应用透视变换，使用智能边界填充"""
    height, width = image.shape[:2]

    # 定义源点（图像四个角）
    src_points = np.float32([[0, 0], [width, 0], [width, height], [0, height]])

    # 随机生成目标点，模拟透视效果 - 适度增强训练数据
    max_offset = int(min(width, height) * stretch_factor * 0.6)  # 从0.3增强到0.6

    dst_points = np.float32([
        [random.randint(0, max_offset), random.randint(0, max_offset)],
        [width - random.randint(0, max_offset), random.randint(0, max_offset)],
        [width - random.randint(0, max_offset), height - random.randint(0, max_offset)],
        [random.randint(0, max_offset), height - random.randint(0, max_offset)]
    ])

    # 计算透视变换矩阵
    perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)

    # 使用智能边界填充
    result = apply_smart_border_fill(image, perspective_matrix, (width, height), is_perspective=True)
    return result

def apply_shear_transform(image, stretch_factor):
    """应用剪切变换，使用智能边界填充"""
    height, width = image.shape[:2]
    center = (width // 2, height // 2)

    # 生成剪切参数 - 适度增强训练数据
    shear_x = stretch_factor * random.uniform(-0.4, 0.4)  # 从±0.2增强到±0.4
    shear_y = stretch_factor * random.uniform(-0.4, 0.4)

    # 创建剪切变换矩阵
    shear_matrix = np.float32([
        [1, shear_x, 0],
        [shear_y, 1, 0]
    ])

    # 使用智能边界填充
    result = apply_smart_border_fill(image, shear_matrix, (width, height), is_perspective=False)
    return result

def apply_scale_transform(image, stretch_factor):
    """应用缩放变换，使用智能边界填充"""
    height, width = image.shape[:2]
    center = (width // 2, height // 2)

    # 生成缩放参数 - 适度增强训练数据
    scale_x = 1 + stretch_factor * random.uniform(-0.5, 0.5)  # 从±0.3增强到±0.5
    scale_y = 1 + stretch_factor * random.uniform(-0.5, 0.5)

    # 确保缩放比例在合理范围内
    scale_x = max(0.6, min(1.6, scale_x))  # 从0.7-1.4扩展到0.6-1.6
    scale_y = max(0.6, min(1.6, scale_y))

    # 创建缩放变换矩阵
    scale_matrix = np.float32([
        [scale_x, 0, center[0] * (1 - scale_x)],
        [0, scale_y, center[1] * (1 - scale_y)]
    ])

    # 使用智能边界填充
    result = apply_smart_border_fill(image, scale_matrix, (width, height), is_perspective=False)
    return result

def apply_smart_border_fill(image, transform_matrix, output_size, is_perspective=False):
    """智能边界填充，使用白色填充"""
    height, width = image.shape[:2]

    # 使用白色填充
    white_color = (255, 255, 255) if len(image.shape) == 3 else 255

    if is_perspective:
        # 透视变换
        result = cv2.warpPerspective(image, transform_matrix, output_size,
                                   borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=white_color)
    else:
        # 仿射变换
        result = cv2.warpAffine(image, transform_matrix, output_size,
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=white_color)

    # 检测并修复边界区域，使用白色
    result = fix_border_artifacts(result, image, white_color)

    return result

def fix_border_artifacts(result, original, fill_color):
    """修复边界伪影，使用白色渐变过渡"""
    height, width = result.shape[:2]

    # 创建掩码，检测白色填充区域
    white_color = (255, 255, 255) if len(result.shape) == 3 else 255

    # 计算每个像素与白色的距离
    if len(result.shape) == 3:
        diff = np.linalg.norm(result - white_color, axis=2)
    else:
        diff = np.abs(result - white_color)

    # 创建边界掩码（接近白色的像素）
    border_mask = diff < 10

    # 对边界区域进行自然过渡处理
    if np.any(border_mask):
        # 创建距离变换，用于渐变
        distance = cv2.distanceTransform(border_mask.astype(np.uint8), cv2.DIST_L2, 5)
        distance = np.clip(distance / 15.0, 0, 1)  # 归一化，更小的过渡区域

        # 对原图进行轻微模糊
        blurred = cv2.GaussianBlur(result, (9, 9), 0)

        # 在边界区域混合模糊和原图，保持白色填充
        for c in range(result.shape[-1] if len(result.shape) == 3 else 1):
            if len(result.shape) == 3:
                # 只在非白色区域应用模糊过渡
                mask = ~border_mask
                result[:, :, c] = np.where(mask,
                                         result[:, :, c] * (1 - distance) + blurred[:, :, c] * distance,
                                         255)  # 保持白色
            else:
                mask = ~border_mask
                result = np.where(mask,
                                result * (1 - distance) + blurred * distance,
                                255)  # 保持白色

    return result

# ---------- 光斑生成函数 ----------
def add_trapezoid_laser(image_path, dot_intensity, glow_size_range, center_brightness, r, g, b, use_gaussian_blur, alpha=1.0, enable_vehicle_effect=True):
    """生成梯形光斑，支持透明度和车辆驶向效果"""
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Unable to load image.")
        return None

    # 根据参数决定是否应用车辆驶向效果
    if enable_vehicle_effect:
        image = apply_random_stretch(image, stretch_range=(0.1, 0.5))  # 适度增强训练数据变形强度
        if image is None:
            print("Error: Failed to apply vehicle approach transformation.")
            return None

    # 如果alpha为0，直接返回处理后的原图
    if alpha <= 0:
        return image.copy()

    height, width = image.shape[:2]
    min_side = min(height, width)

    trapezoid_width = random.randint(int(min_side // 2.5), int(min_side // 1.0))
    trapezoid_height = random.randint(int(min_side // 2.5), int(min_side // 1.0))
    trapezoid_shift = random.randint(-trapezoid_width // 4, trapezoid_width // 4)
    center_x = random.randint(trapezoid_width // 2, width - trapezoid_width // 2)
    center_y = random.randint(trapezoid_height // 2, height - trapezoid_height // 2)

    tl = (center_x - trapezoid_width // 2 + trapezoid_shift, center_y - trapezoid_height // 2)
    tr = (center_x + trapezoid_width // 2 + trapezoid_shift, center_y - trapezoid_height // 2)
    br = (center_x + trapezoid_width // 2, center_y + trapezoid_height // 2)
    bl = (center_x - trapezoid_width // 2, center_y + trapezoid_height // 2)
    pts = np.array([tl, tr, br, bl], dtype=np.int32)

    # 创建光斑mask，使用纯RGB颜色
    mask = np.zeros_like(image, dtype=np.uint8)
    light_color = (b, g, r)  # BGR格式，使用纯RGB颜色
    cv2.fillPoly(mask, [pts], light_color)

    if use_gaussian_blur:
        glow_size = random.randint(glow_size_range[0], glow_size_range[1])
        glow_size += 1 if glow_size % 2 == 0 else 0
        mask = cv2.GaussianBlur(mask, (glow_size, glow_size), 0)

        # 高斯模糊后需要增强颜色强度，补偿模糊造成的颜色稀释
        # 找到mask中的非零区域并增强
        mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        non_zero_mask = mask_gray > 0
        if np.sum(non_zero_mask) > 0:
            # 计算增强因子，基于模糊程度
            enhancement_factor = min(3.0, glow_size / 20.0 + 1.5)
            mask = mask.astype(np.float32)
            mask[non_zero_mask] *= enhancement_factor
            mask = np.clip(mask, 0, 255).astype(np.uint8)

    # 只在有光斑的区域进行混合，其他区域保持原图不变
    result = image.copy()

    # 找到光斑区域（mask中非零的区域）
    mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    light_area = mask_gray > 0

    if np.sum(light_area) > 0:
        # 只在光斑区域进行alpha混合
        result[light_area] = (image[light_area] * (1.0 - alpha) + mask[light_area] * alpha).astype(np.uint8)

    return result

def add_oval_laser(image_path, dot_intensity, glow_size_range, center_brightness, r, g, b, use_gaussian_blur=True, alpha=1.0, enable_vehicle_effect=True):
    """生成椭圆光斑，支持透明度和车辆驶向效果"""
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Unable to load image.")
        return None

    # 根据参数决定是否应用车辆驶向效果
    if enable_vehicle_effect:
        image = apply_random_stretch(image, stretch_range=(0.1, 0.5))  # 适度增强训练数据变形强度
        if image is None:
            print("Error: Failed to apply vehicle approach transformation.")
            return None

    # 如果alpha为0，直接返回处理后的原图
    if alpha <= 0:
        return image.copy()

    height, width = image.shape[:2]
    min_side = min(height, width)

    ellipse_width = random.randint(int(min_side // 2.5), int(min_side // 1.0))
    ellipse_height = random.randint(int(min_side // 2.5), int(min_side // 1.0))
    angle = random.randint(0, 180)
    center_x = random.randint(ellipse_width // 2, width - ellipse_width // 2)
    center_y = random.randint(ellipse_height // 2, height - ellipse_height // 2)

    # 创建光斑mask，使用纯RGB颜色
    mask = np.zeros_like(image, dtype=np.uint8)
    light_color = (b, g, r)  # BGR格式，使用纯RGB颜色
    cv2.ellipse(mask, (center_x, center_y), (ellipse_width // 2, ellipse_height // 2), angle, 0, 360, light_color, -1)

    if use_gaussian_blur:
        glow_size = random.randint(glow_size_range[0], glow_size_range[1])
        glow_size += 1 if glow_size % 2 == 0 else 0
        mask = cv2.GaussianBlur(mask, (glow_size, glow_size), 0)

        # 高斯模糊后需要增强颜色强度，补偿模糊造成的颜色稀释
        mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        non_zero_mask = mask_gray > 0
        if np.sum(non_zero_mask) > 0:
            enhancement_factor = min(3.0, glow_size / 20.0 + 1.5)
            mask = mask.astype(np.float32)
            mask[non_zero_mask] *= enhancement_factor
            mask = np.clip(mask, 0, 255).astype(np.uint8)

    # 只在有光斑的区域进行混合，其他区域保持原图不变
    result = image.copy()

    # 找到光斑区域（mask中非零的区域）
    mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    light_area = mask_gray > 0

    if np.sum(light_area) > 0:
        # 只在光斑区域进行alpha混合
        result[light_area] = (image[light_area] * (1.0 - alpha) + mask[light_area] * alpha).astype(np.uint8)

    return result

def add_single_laser_dot(image_path, dot_intensity, glow_size_range, center_brightness, r, g, b, alpha=1.0):
    """生成单个光点（光晕），支持透明度"""
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Unable to load image.")
        return None

    # 对原始图像应用随机拉伸变形
    image = apply_random_stretch(image, stretch_range=(0.1, 0.4))  # 适度增强训练数据变形强度
    if image is None:
        print("Error: Failed to apply stretch transformation.")
        return None

    # 如果alpha为0，直接返回拉伸后的原图
    if alpha <= 0:
        return image.copy()

    height, width = image.shape[:2]
    min_xy = min(height, width)

    dot_radius = random.randint(int(min_xy // 3), int(min_xy // 1.5))
    dot_place_x = random.randint(width // 5, width // 5)
    dot_place_y = random.randint(height // 5, height // 5)
    center_x = random.randint(dot_place_x, width - dot_place_x)
    center_y = random.randint(dot_place_y, height - dot_place_y)

    # 创建光斑mask，使用纯RGB颜色
    mask = np.zeros_like(image, dtype=np.uint8)
    light_color = (b, g, r)  # BGR格式，使用纯RGB颜色
    cv2.circle(mask, (center_x, center_y), dot_radius, light_color, -1)

    glow_size = random.randint(glow_size_range[0], glow_size_range[1])
    glow_size += 1 if glow_size % 2 == 0 else 0
    mask = cv2.GaussianBlur(mask, (glow_size, glow_size), 0)

    # 高斯模糊后需要增强颜色强度，补偿模糊造成的颜色稀释
    mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    non_zero_mask = mask_gray > 0
    if np.sum(non_zero_mask) > 0:
        enhancement_factor = min(3.0, glow_size / 20.0 + 1.5)
        mask = mask.astype(np.float32)
        mask[non_zero_mask] *= enhancement_factor
        mask = np.clip(mask, 0, 255).astype(np.uint8)

    # 只在有光斑的区域进行混合，其他区域保持原图不变
    result = image.copy()

    # 找到光斑区域（mask中非零的区域）
    mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    light_area = mask_gray > 0

    if np.sum(light_area) > 0:
        # 只在光斑区域进行alpha混合
        result[light_area] = (image[light_area] * (1.0 - alpha) + mask[light_area] * alpha).astype(np.uint8)

    return result

# ---------- 核心生成函数 ----------
def generate(num, image_path, dot_intensity, glow_size_range, center_brightness, r, g, b, type_light, flag, alpha=1.0, enable_vehicle_effect=True):
    """生成 num 张光斑图，梯形+椭圆，集成车辆驶向效果"""
    output_dir = os.path.join("jiguang", str(type_light))
    os.makedirs(output_dir, exist_ok=True)

    i = 0
    while i < num // 2:
        output_image = add_trapezoid_laser(image_path, dot_intensity, glow_size_range, center_brightness, r, g, b, flag, alpha, enable_vehicle_effect)
        if output_image is not None:
            parts = image_path.replace("\\","/").split("/")  # 支持中英文路径
            number = os.path.splitext(parts[-1])[0]
            cv2.imwrite(os.path.join(output_dir, f"{number}_{type_light}_{i}.jpg"), output_image)
            i += 1
        else:
            break

    while i < num:
        output_image = add_oval_laser(image_path, dot_intensity, glow_size_range, center_brightness, r, g, b, flag, alpha, enable_vehicle_effect)
        if output_image is not None:
            parts = image_path.replace("\\","/").split("/")
            number = os.path.splitext(parts[-1])[0]
            cv2.imwrite(os.path.join(output_dir, f"{number}_{type_light}_{i}.jpg"), output_image)
            i += 1
        else:
            break


def generate_0(image_path, num, glow_size_range, center_brightness, enable_vehicle_effect=True):
    """
    生成不同颜色、强度的光斑序列，集成车辆驶向效果

    Args:
        image_path: 输入图像路径
        num: 每种类型生成的图片数量
        glow_size_range: 光斑大小范围
        center_brightness: 中心亮度
        enable_vehicle_effect: 是否启用车辆驶向效果（默认True）
    """
    type_light = 0

    # 定义颜色列表
    colors = [
        (175, 0, 175), (185, 0, 185), (195, 0, 195), (205, 0, 205), (215, 0, 215),
        (225, 0, 225), (235, 0, 235), (245, 0, 245), (255, 0, 255),
        # 紫色
        (0, 175, 0), (0, 185, 0), (0, 195, 0), (0, 205, 0), (0, 215, 0),
        (0, 225, 0), (0, 235, 0), (0, 245, 0), (0, 255, 0),
        # 绿色
        (175, 0, 0), (185, 0, 0), (195, 0, 0), (205, 0, 0), (215, 0, 0),
        (225, 0, 0), (235, 0, 0), (245, 0, 0), (255, 0, 0)
        # 红色
    ]
    colors_1 = [
        (240, 240, 100),  # 黄
        (255, 255, 255)  # 白
    ]

    # dot_i_list 控制透明度，从高到低
    dot_i_list = [1]
    dot_i_list_1 = [1.0, 0.92, 0.85, 0.78, 0.7, 0.62, 0.74, 0.68, 0.6]


    for color_idx, (r, g, b) in enumerate(colors):
        for intensity_idx, alpha in enumerate(dot_i_list):
            dot_intensity = 1.0  # dot_intensity 设置为1.0，透明度完全由alpha控制
            # 传递车辆效果参数到generate函数
            generate(num, image_path, dot_intensity, glow_size_range, center_brightness,
                    r, g, b, type_light, flag=False, alpha=alpha,
                    enable_vehicle_effect=enable_vehicle_effect)
            type_light += 1
    # 为每种颜色的每个透明度级别分配一个唯一的type_light
    for color_idx, (r, g, b) in enumerate(colors_1):
        for intensity_idx, alpha in enumerate(dot_i_list_1):
            dot_intensity = 1.0  # dot_intensity 设置为1.0，透明度完全由alpha控制
            # 传递车辆效果参数到generate函数
            generate(num, image_path, dot_intensity, glow_size_range, center_brightness,
                    r, g, b, type_light, flag=False, alpha=alpha,
                    enable_vehicle_effect=enable_vehicle_effect)
            type_light += 1

    # 最后生成一个完全透明的（alpha=0，仅应用车辆效果）
    print(f"🎨 正在生成透明图片 (type_light {type_light})...")
    generate(num, image_path, 1.0, glow_size_range, center_brightness,
            255, 255, 255, type_light, flag=False, alpha=0,
            enable_vehicle_effect=enable_vehicle_effect)

    print(f"✅ 生成完成！总共 {type_light + 1} 种类型，每种 {num} 张图片")


def convert():
    path_jiguang = 'jiguang'
    path_test = 'classification-jiguang'
    os.makedirs(path_test, exist_ok=True)
    class_num = 14
    # Create target directory structure
    for i in range(1, class_num + 1):
        os.makedirs(os.path.join(path_test, str(i)), exist_ok=True)

    # Traverse source directory
    for root, dirs, files in os.walk(path_jiguang):
        for file in files:
            if file.endswith('.jpg'):
                # Parse filename
                parts = file.split('_')
                if len(parts) >= 1 and parts[0].isdigit():
                    folder_num = int(parts[0])
                    if 1 <= folder_num <= class_num:
                        # Source and destination paths
                        src_path = os.path.join(root, file)
                        dest_path = os.path.join(path_test, str(folder_num), file)

                        # Copy file
                        shutil.copy2(src_path, dest_path)
                        print(f'Copied {src_path} to {dest_path}')
                    else:
                        print(f'Skipping {file}: folder number out of range (1-10)')
                else:
                    print(f'Skipping {file}: invalid naming format')

    print("File classification completed!")

import cv2
import numpy as np
import random
import os

def _make_odd(x):
    x = max(1, int(x))
    return x if x % 2 == 1 else x - 1 if x > 1 else 1

def generate_moving_laser_sequence_old(image_path, output_dir, glow_size_range, center_brightness=255):
    """
    生成 60 张图（5 种干扰光，每种 12 张，光晕大小在组内固定，位置线性移动）
    - 不会平移或变形原图，只平移遮罩（避免“路牌被改变”问题）
    - glow_size 会依据图片尺寸自动限制，避免覆盖整个图像
    """
    # 使用与generate_0完全相同的颜色定义
    colors = [
        (255, 120, 100),  # 红
        (38, 219, 111),   # 绿
        (80, 150, 250),   # 蓝
        (240, 240, 100),  # 黄
        (255, 255, 255)   # 白
    ]

    # 使用与generate_0相同的透明度列表（选择5个代表性透明度）
    selected_alphas = [1.0, 0.85, 0.7, 0.6, 0.5]  # 从9个透明度中选择5个

    # 安全处理 brightness（映射到 0-255）
    center_brightness = int(center_brightness)
    if center_brightness < 0:
        center_brightness = 0
    if center_brightness > 255:
        center_brightness = 255

    # load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法加载图片: {image_path}")

    h, w = image.shape[:2]
    min_side = min(h, w)
    os.makedirs(output_dir, exist_ok=True)
    img_name = os.path.splitext(os.path.basename(image_path))[0]

    num_per_light = 12  # 每种干扰 12 张 -> 5*12 = 60

    print(f"开始生成移动光斑序列，共5组，每组12张...")

    for idx, ((r, g, b), alpha) in enumerate(zip(colors, selected_alphas)):
        color_names = ["红色", "绿色", "蓝色", "黄色", "白色"]
        print(f"正在生成第 {idx+1} 组 ({color_names[idx]} 光斑, 透明度={alpha})...")

        # 每组选定一种形状并在该组内保持一致
        shape_type = random.choice(['trapezoid', 'oval'])

        # --- 固定组内的 glow_size（但对给定 glow_size_range 做合理裁剪） ---
        # 从用户提供区间取一个 candidate（如果区间不合理就忽略）
        if isinstance(glow_size_range, (tuple, list)) and len(glow_size_range) == 2 and glow_size_range[0] <= glow_size_range[1]:
            glow_cand = random.randint(glow_size_range[0], glow_size_range[1])
        else:
            glow_cand = int(min_side * 0.2)

        # 限制 glow 的上限为图像尺寸的比例，避免模糊覆盖全图
        glow_upper = max(3, int(min_side * 0.25))  # 建议最大不要超过 min_side * 0.25
        glow_size = min(glow_cand, glow_upper)
        glow_size = _make_odd(glow_size)

        # 使用与generate_0完全相同的尺寸计算方式
        if shape_type == 'trapezoid':
            # 与add_trapezoid_laser函数相同的尺寸计算
            trapezoid_width = random.randint(int(min_side // 2.5), int(min_side // 1.0))
            trapezoid_height = random.randint(int(min_side // 2.5), int(min_side // 1.0))
            trapezoid_shift = random.randint(-trapezoid_width // 4, trapezoid_width // 4)
        else:
            # 与add_oval_laser函数相同的尺寸计算
            ellipse_width = random.randint(int(min_side // 2.5), int(min_side // 1.0))
            ellipse_height = random.randint(int(min_side // 2.5), int(min_side // 1.0))
            angle = random.randint(0, 180)

        # --- 产生一条连贯的移动轨迹（起点、终点在图片安全边界内） ---
        margin_x = max(5, w // 8)
        margin_y = max(5, h // 8)
        start_x = random.randint(margin_x, w - margin_x)
        start_y = random.randint(margin_y, h - margin_y)
        end_x = random.randint(margin_x, w - margin_x)
        end_y = random.randint(margin_y, h - margin_y)

        # 如果起点与终点太接近，稍微强制拉开（保证可见移动）
        if abs(start_x - end_x) + abs(start_y - end_y) < min_side // 20:
            end_x = min(max(margin_x, end_x + (min_side // 6)), w - margin_x)
            end_y = min(max(margin_y, end_y + (min_side // 6)), h - margin_y)

        # 组内 12 帧
        for step in range(num_per_light):
            t = step / (num_per_light - 1) if num_per_light > 1 else 0.0
            cx = int(round(start_x * (1 - t) + end_x * t))
            cy = int(round(start_y * (1 - t) + end_y * t))

            # 使用与训练数据完全相同的光斑生成方式
            mask = np.zeros_like(image, dtype=np.uint8)
            # 使用纯RGB颜色，与add_trapezoid_laser/add_oval_laser相同
            light_color = (b, g, r)  # BGR格式

            if shape_type == 'trapezoid':
                # 使用与add_trapezoid_laser相同的顶点计算方式
                tl = (cx - trapezoid_width // 2 + trapezoid_shift, cy - trapezoid_height // 2)
                tr = (cx + trapezoid_width // 2 + trapezoid_shift, cy - trapezoid_height // 2)
                br = (cx + trapezoid_width // 2, cy + trapezoid_height // 2)
                bl = (cx - trapezoid_width // 2, cy + trapezoid_height // 2)
                pts = np.array([tl, tr, br, bl], dtype=np.int32)
                cv2.fillPoly(mask, [pts], light_color)
            else:
                # 使用与add_oval_laser相同的椭圆绘制方式
                cv2.ellipse(mask, (cx, cy), (ellipse_width // 2, ellipse_height // 2), angle, 0, 360, light_color, -1)

            # 使用与训练数据相同的高斯模糊
            mask_blur = cv2.GaussianBlur(mask, (glow_size, glow_size), 0)

            # 使用与训练数据相同的alpha混合方式
            result = image.copy()
            mask_gray = cv2.cvtColor(mask_blur, cv2.COLOR_BGR2GRAY)
            light_area = mask_gray > 0

            if np.sum(light_area) > 0:
                # 只在光斑区域进行alpha混合
                result[light_area] = (image[light_area] * (1.0 - alpha) + mask_blur[light_area] * alpha).astype(np.uint8)

            out_name = f"{img_name}_moving_light{idx+1}_{step+1}.jpg"
            cv2.imwrite(os.path.join(output_dir, out_name), result)

    print(f"生成完成：60 张图保存在 {output_dir}")


def generate_moving_deformation_sequence_old(image_path, output_dir, glow_size_range, center_brightness=255):
    """
    第二个函数：生成空间变形序列，与训练数据生成方式完全一致
    - 使用与generate_0相同的颜色、透明度、尺寸计算、混合方式
    - 每组随机选取一种变形方式（透视、剪切、缩放）
    - 空间扭曲效果连续变化，光斑位置固定
    - 确保测试集与训练集数据分布一致，避免域偏移问题
    """
    # 使用与generate_0完全相同的颜色定义
    colors = [
        (255, 120, 100),  # 红
        (38, 219, 111),   # 绿
        (80, 150, 250),   # 蓝
        (240, 240, 100),  # 黄
        (255, 255, 255)   # 白
    ]

    # 使用与generate_0相同的透明度列表（选择5个代表性透明度）
    selected_alphas = [1.0, 0.85, 0.7, 0.6, 0.5]

    # 安全处理 brightness
    center_brightness = int(center_brightness)
    center_brightness = max(0, min(255, center_brightness))

    # 加载图像
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法加载图片: {image_path}")

    h, w = image.shape[:2]
    min_side = min(h, w)
    os.makedirs(output_dir, exist_ok=True)
    img_name = os.path.splitext(os.path.basename(image_path))[0]

    num_per_light = 12  # 每种干扰 12 张 -> 5*12 = 60

    # 定义三种变形策略
    deformation_strategies = ['perspective', 'shear', 'scale']

    for idx, ((r, g, b), alpha) in enumerate(zip(colors, selected_alphas)):
        # 每组随机选择一种变形方式
        strategy = random.choice(deformation_strategies)

        # 固定光斑位置（每组内保持一致）
        laser_x = random.randint(w // 4, 3 * w // 4)
        laser_y = random.randint(h // 4, 3 * h // 4)

        # 固定光斑形状和大小
        shape_type = random.choice(['trapezoid', 'oval'])

        # 固定 glow_size
        if isinstance(glow_size_range, (tuple, list)) and len(glow_size_range) == 2:
            glow_cand = random.randint(glow_size_range[0], glow_size_range[1])
        else:
            glow_cand = int(min_side * 0.2)

        glow_upper = max(3, int(min_side * 0.25))
        glow_size = min(glow_cand, glow_upper)
        glow_size = _make_odd(glow_size)

        # 使用与generate_0完全相同的尺寸计算方式
        if shape_type == 'trapezoid':
            # 与add_trapezoid_laser函数相同的尺寸计算
            trapezoid_width = random.randint(int(min_side // 2.5), int(min_side // 1.0))
            trapezoid_height = random.randint(int(min_side // 2.5), int(min_side // 1.0))
            trapezoid_shift = random.randint(-trapezoid_width // 4, trapezoid_width // 4)
        else:
            # 与add_oval_laser函数相同的尺寸计算
            ellipse_width = random.randint(int(min_side // 2.5), int(min_side // 1.0))
            ellipse_height = random.randint(int(min_side // 2.5), int(min_side // 1.0))
            angle = random.randint(0, 180)

        # 定义变形参数的起始和结束值，实现连续变化
        if strategy == 'perspective':
            # 透视变换：从轻微到明显的透视效果
            start_factor = 0.05
            end_factor = 0.35
        elif strategy == 'shear':
            # 剪切变换：从无剪切到明显剪切
            start_factor = 0.0
            end_factor = 0.3
        else:  # scale
            # 缩放变换：从正常到明显缩放
            start_factor = 0.05
            end_factor = 0.4

        print(f"生成第 {idx+1} 组 ({strategy} 变形)...")

        # 组内 12 帧，变形效果连续变化
        for step in range(num_per_light):
            # 计算当前帧的变形强度（线性插值）
            t = step / (num_per_light - 1) if num_per_light > 1 else 0.0
            current_factor = start_factor * (1 - t) + end_factor * t

            # 应用空间变形
            deformed_image = apply_continuous_deformation(image, strategy, current_factor)

            # 使用与训练数据完全相同的光斑生成方式
            mask = np.zeros_like(deformed_image, dtype=np.uint8)
            # 使用纯RGB颜色，与add_trapezoid_laser/add_oval_laser相同
            light_color = (b, g, r)  # BGR格式

            if shape_type == 'trapezoid':
                # 使用与add_trapezoid_laser相同的顶点计算方式
                tl = (laser_x - trapezoid_width // 2 + trapezoid_shift, laser_y - trapezoid_height // 2)
                tr = (laser_x + trapezoid_width // 2 + trapezoid_shift, laser_y - trapezoid_height // 2)
                br = (laser_x + trapezoid_width // 2, laser_y + trapezoid_height // 2)
                bl = (laser_x - trapezoid_width // 2, laser_y + trapezoid_height // 2)
                pts = np.array([tl, tr, br, bl], dtype=np.int32)
                cv2.fillPoly(mask, [pts], light_color)
            else:
                # 使用与add_oval_laser相同的椭圆绘制方式
                cv2.ellipse(mask, (laser_x, laser_y), (ellipse_width // 2, ellipse_height // 2), angle, 0, 360, light_color, -1)

            # 使用与训练数据相同的高斯模糊
            mask_blur = cv2.GaussianBlur(mask, (glow_size, glow_size), 0)

            # 使用与训练数据相同的alpha混合方式
            result = deformed_image.copy()
            mask_gray = cv2.cvtColor(mask_blur, cv2.COLOR_BGR2GRAY)
            light_area = mask_gray > 0

            if np.sum(light_area) > 0:
                # 只在光斑区域进行alpha混合
                result[light_area] = (deformed_image[light_area] * (1.0 - alpha) + mask_blur[light_area] * alpha).astype(np.uint8)

            out_name = f"{img_name}_deform_{strategy}_light{idx+1}_{step+1}.jpg"
            cv2.imwrite(os.path.join(output_dir, out_name), result)

    print(f"生成完成：60 张变形图保存在 {output_dir}")


def apply_strong_deformation(image, strategy, factor):
    """
    应用强变形，专门用于测试数据生成
    """
    if factor <= 0:
        return image.copy()

    height, width = image.shape[:2]

    if strategy == 'perspective':
        return apply_strong_perspective(image, factor)
    elif strategy == 'shear':
        return apply_strong_shear(image, factor)
    else:  # scale
        return apply_strong_scale(image, factor)


def apply_strong_perspective(image, factor):
    """强透视变换，专门用于测试数据"""
    height, width = image.shape[:2]

    # 定义源点
    src_points = np.float32([[0, 0], [width, 0], [width, height], [0, height]])

    # 温和但明显的变形参数
    max_offset = int(min(width, height) * factor * 1.0)  # 调整到1.0

    dst_points = np.float32([
        [max_offset * 0.5, max_offset * 0.4],  # 温和的偏移系数
        [width - max_offset * 0.4, max_offset * 0.5],
        [width - max_offset * 0.5, height - max_offset * 0.4],
        [max_offset * 0.4, height - max_offset * 0.5]
    ])

    # 计算透视变换矩阵
    perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)

    # 应用变换
    result = cv2.warpPerspective(image, perspective_matrix, (width, height),
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=(255, 255, 255))
    return result


def apply_strong_shear(image, factor):
    """强剪切变换，专门用于测试数据"""
    height, width = image.shape[:2]

    # 温和但明显的变形参数
    shear_x = factor * 0.8  # 调整到0.8
    shear_y = factor * 0.6  # 调整到0.6

    # 创建剪切变换矩阵
    shear_matrix = np.float32([
        [1, shear_x, 0],
        [shear_y, 1, 0]
    ])

    # 应用变换
    result = cv2.warpAffine(image, shear_matrix, (width, height),
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=(255, 255, 255))
    return result


def apply_strong_scale(image, factor):
    """强缩放变换，专门用于测试数据"""
    height, width = image.shape[:2]
    center = (width // 2, height // 2)

    # 温和但明显的变形参数
    scale_x = 1 + factor * 0.7  # 调整到0.7
    scale_y = 1 - factor * 0.5  # 调整到0.5

    # 合理的变形范围
    scale_x = max(0.4, min(2.0, scale_x))  # 调整到0.4-2.0
    scale_y = max(0.4, min(2.0, scale_y))

    # 创建缩放变换矩阵
    scale_matrix = np.float32([
        [scale_x, 0, center[0] * (1 - scale_x)],
        [0, scale_y, center[1] * (1 - scale_y)]
    ])

    # 应用变换
    result = cv2.warpAffine(image, scale_matrix, (width, height),
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=(255, 255, 255))
    return result


def apply_continuous_deformation(image, strategy, factor):
    """
    应用连续的空间变形，用于模拟车辆行驶效果

    Args:
        image: 输入图像
        strategy: 变形策略 ('perspective', 'shear', 'scale')
        factor: 变形强度因子 (0.0 到 1.0)

    Returns:
        变形后的图像
    """
    if factor <= 0:
        return image.copy()

    height, width = image.shape[:2]

    if strategy == 'perspective':
        # 连续透视变换
        return apply_continuous_perspective(image, factor)
    elif strategy == 'shear':
        # 连续剪切变换
        return apply_continuous_shear(image, factor)
    else:  # scale
        # 连续缩放变换
        return apply_continuous_scale(image, factor)


def apply_continuous_perspective(image, factor):
    """连续透视变换"""
    height, width = image.shape[:2]

    # 定义源点
    src_points = np.float32([[0, 0], [width, 0], [width, height], [0, height]])

    # 根据factor计算偏移量，实现连续变化
    max_offset = int(min(width, height) * factor * 0.4)

    # 使用固定的随机种子确保同一组内变形方向一致
    np.random.seed(hash(str(factor)) % 2**32)

    dst_points = np.float32([
        [max_offset * 0.3, max_offset * 0.2],
        [width - max_offset * 0.2, max_offset * 0.3],
        [width - max_offset * 0.3, height - max_offset * 0.2],
        [max_offset * 0.2, height - max_offset * 0.3]
    ])

    # 计算透视变换矩阵
    perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)

    # 应用变换
    result = cv2.warpPerspective(image, perspective_matrix, (width, height),
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=(255, 255, 255))
    return result


def apply_continuous_shear(image, factor):
    """连续剪切变换"""
    height, width = image.shape[:2]

    # 根据factor计算剪切参数
    shear_x = factor * 0.3
    shear_y = factor * 0.2

    # 创建剪切变换矩阵
    shear_matrix = np.float32([
        [1, shear_x, 0],
        [shear_y, 1, 0]
    ])

    # 应用变换
    result = cv2.warpAffine(image, shear_matrix, (width, height),
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=(255, 255, 255))
    return result


def apply_continuous_scale(image, factor):
    """连续缩放变换"""
    height, width = image.shape[:2]
    center = (width // 2, height // 2)

    # 根据factor计算缩放参数
    scale_x = 1 + factor * 0.4
    scale_y = 1 - factor * 0.2

    # 创建缩放变换矩阵
    scale_matrix = np.float32([
        [scale_x, 0, center[0] * (1 - scale_x)],
        [0, scale_y, center[1] * (1 - scale_y)]
    ])

    # 应用变换
    result = cv2.warpAffine(image, scale_matrix, (width, height),
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=(255, 255, 255))
    return result


def generate_moving_laser_sequence(image_path, output_dir, glow_size_range, center_brightness=255):
    """
    第一个函数：生成移动纯色光斑序列
    - 使用纯色RGB光斑，不使用高斯模糊
    - 光斑位置连续移动，无空间变形
    - 使用不同透明度的纯色覆盖
    """
    # 使用纯色RGB颜色
    colors = [
        (255, 120, 100),  # 红
        (38, 219, 111),   # 绿
        (80, 150, 250),   # 蓝
        (240, 240, 100),  # 黄
        (255, 255, 255)   # 白
    ]

    # 不同透明度
    selected_alphas = [1, 0.85, 0.92, 1, 1]

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    img_name = os.path.splitext(os.path.basename(image_path))[0]

    # 加载原图
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法加载图片: {image_path}")

    h, w = image.shape[:2]
    min_side = min(h, w)

    print(f"开始生成移动纯色光斑序列，共5组，每组12张...")

    # 为每种颜色生成12张移动光斑图
    for idx, ((r, g, b), alpha) in enumerate(zip(colors, selected_alphas)):
        color_names = ["红色", "绿色", "蓝色", "黄色", "紫色"]
        print(f"正在生成第 {idx+1} 组 ({color_names[idx]} 光斑, 透明度={alpha})...")

        # 使用与generate_0完全相同的光斑大小计算
        if idx % 2 == 0:
            # 梯形光斑 - 与add_trapezoid_laser完全相同
            shape_w = random.randint(int(min_side // 2.5), int(min_side // 1.0))
            shape_h = random.randint(int(min_side // 2.5), int(min_side // 1.0))
            shape_shift = random.randint(-shape_w // 4, shape_w // 4)  # 与训练数据相同
            shape_type = 'trapezoid'
        else:
            # 椭圆光斑 - 与add_oval_laser完全相同
            ellipse_w = random.randint(int(min_side // 2.5), int(min_side // 1.0))
            ellipse_h = random.randint(int(min_side // 2.5), int(min_side // 1.0))
            angle = random.randint(0, 180)
            shape_type = 'ellipse'

        # 定义移动轨迹
        margin_x = max(5, w // 8)
        margin_y = max(5, h // 8)
        start_x = random.randint(margin_x, w - margin_x)
        start_y = random.randint(margin_y, h - margin_y)
        end_x = random.randint(margin_x, w - margin_x)
        end_y = random.randint(margin_y, h - margin_y)

        # 生成12张图，光斑位置连续移动
        for step in range(12):
            # 计算当前位置
            t = step / 11.0 if step < 11 else 1.0
            cx = int(start_x * (1 - t) + end_x * t)
            cy = int(start_y * (1 - t) + end_y * t)

            # 创建纯色光斑
            result = image.copy()

            if shape_type == 'trapezoid':
                # 梯形顶点
                tl = (cx - shape_w // 2 + shape_shift, cy - shape_h // 2)
                tr = (cx + shape_w // 2 + shape_shift, cy - shape_h // 2)
                br = (cx + shape_w // 2, cy + shape_h // 2)
                bl = (cx - shape_w // 2, cy + shape_h // 2)
                pts = np.array([tl, tr, br, bl], dtype=np.int32)

                # 创建mask
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(mask, [pts], 255)

                # 应用纯色
                light_color = np.array([b, g, r], dtype=np.uint8)
                result[mask > 0] = result[mask > 0] * (1 - alpha) + light_color * alpha

            else:  # ellipse
                # 创建mask
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.ellipse(mask, (cx, cy), (ellipse_w // 2, ellipse_h // 2), angle, 0, 360, 255, -1)

                # 应用纯色
                light_color = np.array([b, g, r], dtype=np.uint8)
                result[mask > 0] = result[mask > 0] * (1 - alpha) + light_color * alpha

            # 保存图片
            out_name = f"{img_name}_moving_light{idx+1}_{step+1}.jpg"
            cv2.imwrite(os.path.join(output_dir, out_name), result.astype(np.uint8))

    print(f"生成完成：60 张移动纯色光斑图保存在 {output_dir}")


def generate_moving_deformation_sequence(image_path, output_dir, glow_size_range, center_brightness=255):
    """
    第二个函数：生成空间变形+纯色光斑序列
    - 使用纯色RGB光斑，不使用高斯模糊
    - 大幅加强空间扭曲效果，模拟车辆行驶
    - 光斑位置固定，使用不同透明度的纯色覆盖
    """
    # 使用纯色RGB颜色
    colors = [
        (255, 120, 100),  # 红
        (38, 219, 111),   # 绿
        (80, 150, 250),   # 蓝
        (240, 240, 100),  # 黄
        (255, 255, 255)   # 白
    ]

    # 不同透明度
    selected_alphas = [1, 0.85, 0.92, 1, 1]

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    img_name = os.path.splitext(os.path.basename(image_path))[0]

    # 加载原图
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法加载图片: {image_path}")

    h, w = image.shape[:2]
    min_side = min(h, w)

    # 定义三种变形策略
    deformation_strategies = ['perspective', 'shear', 'scale']

    print(f"开始生成空间变形+纯色光斑序列，共5组，每组12张...")

    # 为每种颜色生成12张变形图
    for idx, ((r, g, b), alpha) in enumerate(zip(colors, selected_alphas)):
        color_names = ["红色", "绿色", "蓝色", "黄色", "紫色"]
        strategy = random.choice(deformation_strategies)
        print(f"正在生成第 {idx+1} 组 ({color_names[idx]} 光斑, {strategy} 变形, 透明度={alpha})...")

        # 固定光斑位置和大小
        laser_x = random.randint(w // 4, 3 * w // 4)
        laser_y = random.randint(h // 4, 3 * h // 4)

        # 使用与generate_0完全相同的光斑大小计算
        if idx % 2 == 0:
            # 梯形光斑 - 与add_trapezoid_laser完全相同
            shape_w = random.randint(int(min_side // 2.5), int(min_side // 1.0))
            shape_h = random.randint(int(min_side // 2.5), int(min_side // 1.0))
            shape_shift = random.randint(-shape_w // 4, shape_w // 4)  # 与训练数据相同
            shape_type = 'trapezoid'
        else:
            # 椭圆光斑 - 与add_oval_laser完全相同
            ellipse_w = random.randint(int(min_side // 2.5), int(min_side // 1.0))
            ellipse_h = random.randint(int(min_side // 2.5), int(min_side // 1.0))
            angle = random.randint(0, 180)
            shape_type = 'ellipse'

        # 生成12张图，变形程度连续变化
        for step in range(12):
            # 计算变形强度
            t = step / 11.0

            # 增强连续变化的变形强度，从轻微到明显
            # 在12帧内从0.05连续增加到0.8，增加变形的连续性和可见度
            deform_factor = 0.05 + t * 0.75  # t从0到1，变形强度从0.05到0.8

            # 使用连续变形函数，确保真正的连续性
            deformed_image = apply_continuous_deformation(image, strategy, deform_factor)

            # 创建纯色光斑
            result = deformed_image.copy()

            if shape_type == 'trapezoid':
                # 梯形顶点
                tl = (laser_x - shape_w // 2 + shape_shift, laser_y - shape_h // 2)
                tr = (laser_x + shape_w // 2 + shape_shift, laser_y - shape_h // 2)
                br = (laser_x + shape_w // 2, laser_y + shape_h // 2)
                bl = (laser_x - shape_w // 2, laser_y + shape_h // 2)
                pts = np.array([tl, tr, br, bl], dtype=np.int32)

                # 创建mask
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(mask, [pts], 255)

                # 应用纯色
                light_color = np.array([b, g, r], dtype=np.uint8)
                result[mask > 0] = result[mask > 0] * (1 - alpha) + light_color * alpha

            else:  # ellipse
                # 创建mask
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.ellipse(mask, (laser_x, laser_y), (ellipse_w // 2, ellipse_h // 2), angle, 0, 360, 255, -1)

                # 应用纯色
                light_color = np.array([b, g, r], dtype=np.uint8)
                result[mask > 0] = result[mask > 0] * (1 - alpha) + light_color * alpha

            # 保存图片
            out_name = f"{img_name}_deform_{strategy}_light{idx+1}_{step+1}.jpg"
            cv2.imwrite(os.path.join(output_dir, out_name), result.astype(np.uint8))

    print(f"生成完成：60 张变形+纯色光斑图保存在 {output_dir}")


def generate_3d_perspective_sequence(image_path, output_dir="3rd_1", glow_size_range=None, center_brightness=None):
    """
    第三个函数：生成纯3D透视变换序列（无干扰光）
    模拟车辆从远处驶向告示牌，从正面到右侧面的立体空间感

    Args:
        image_path: 输入图像路径
        output_dir: 输出目录（默认为3rd_1）
        glow_size_range: 未使用（保持兼容性）
        center_brightness: 未使用（保持兼容性）
    """
    if not os.path.exists(image_path):
        print(f"错误：找不到图像文件 {image_path}")
        return

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 读取原始图像
    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"错误：无法读取图像 {image_path}")
        return

    # 获取图像基本信息
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    h, w = original_image.shape[:2]

    print("开始生成纯3D透视变换序列，共60张连续变换图片...")
    print("模拟车辆从远到近，从正面视角到右侧面视角")

    total_images = 0

    for frame_idx in range(60):
        # 计算全局时间参数（0到1，跨越整个60张图片）
        global_t = frame_idx / 59.0

        # 应用连续的3D透视变换
        # 模拟车辆从远到近，从正面到右侧面的连续变化
        transformed_image = apply_continuous_3d_transform(
            original_image,
            global_t
        )

        # 保存图像
        filename = f"{base_name}_3d_transform_{frame_idx + 1:02d}.jpg"
        filepath = os.path.join(output_dir, filename)
        cv2.imwrite(filepath, transformed_image)
        total_images += 1

        # 显示进度
        if (frame_idx + 1) % 10 == 0:
            progress_percent = (frame_idx + 1) / 60 * 100
            print(f"  进度: {frame_idx + 1}/60 ({progress_percent:.0f}%)")

    print(f"生成完成：{total_images} 张纯3D透视变换图保存在 {output_dir}")
    print("变换效果：从正面视角(0°)到右侧面视角(45°)，模拟车辆接近")
    return total_images


def add_interference_light(image, position, light_type, color, size, brightness):
    """
    添加不同类型的干扰光

    Args:
        image: 输入图像
        position: 光斑位置 (x_ratio, y_ratio)
        light_type: 光斑类型 ('circular', 'elliptical', 'rectangular', 'triangular', 'star')
        color: 光斑颜色 (B, G, R)
        size: 光斑大小
        brightness: 亮度
    """
    result = image.copy()
    h, w = image.shape[:2]

    # 计算实际位置
    center_x = int(position[0] * w)
    center_y = int(position[1] * h)

    # 创建mask
    mask = np.zeros((h, w), dtype=np.uint8)

    if light_type == "circular":
        # 圆形光斑
        cv2.circle(mask, (center_x, center_y), size // 2, 255, -1)

    elif light_type == "elliptical":
        # 椭圆光斑
        axes = (size // 2, int(size * 0.7) // 2)
        cv2.ellipse(mask, (center_x, center_y), axes, 0, 0, 360, 255, -1)

    elif light_type == "rectangular":
        # 矩形光斑
        half_size = size // 2
        pt1 = (center_x - half_size, center_y - half_size)
        pt2 = (center_x + half_size, center_y + half_size)
        cv2.rectangle(mask, pt1, pt2, 255, -1)

    elif light_type == "triangular":
        # 三角形光斑
        half_size = size // 2
        pts = np.array([
            [center_x, center_y - half_size],           # 顶点
            [center_x - half_size, center_y + half_size], # 左下
            [center_x + half_size, center_y + half_size]  # 右下
        ], dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)

    elif light_type == "star":
        # 星形光斑（简化为菱形）
        half_size = size // 2
        pts = np.array([
            [center_x, center_y - half_size],           # 上
            [center_x + half_size, center_y],           # 右
            [center_x, center_y + half_size],           # 下
            [center_x - half_size, center_y]            # 左
        ], dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)

    # 应用颜色和透明度
    alpha = 0.7  # 固定透明度
    light_color = np.array(color, dtype=np.uint8)
    result[mask > 0] = result[mask > 0] * (1 - alpha) + light_color * alpha

    return result


def apply_3d_perspective_transform(image, progress):
    """
    应用3D透视变换，模拟从正面到右侧面的视角变化

    Args:
        image: 输入图像
        progress: 进度 (0到1)，0为正面视角，1为右侧面视角
    """
    h, w = image.shape[:2]

    # 计算3D透视参数
    # progress = 0: 正面视角
    # progress = 1: 右侧面视角（约45度）

    # 1. Y轴旋转角度（水平旋转）
    y_rotation_angle = progress * 45  # 最大45度

    # 2. 轻微的X轴旋转（垂直倾斜）
    x_rotation_angle = progress * 10  # 最大10度

    # 3. 透视强度
    perspective_strength = progress * 0.3

    # 计算变换矩阵
    # 模拟3D到2D的投影

    # 源点（原图四个角）
    src_points = np.float32([
        [0, 0],
        [w-1, 0],
        [w-1, h-1],
        [0, h-1]
    ])

    # 目标点（3D透视后的四个角）
    # 简化透视变换，确保数值稳定性

    # 右侧收缩，左侧扩展（Y轴旋转效果）
    right_shrink = perspective_strength * w * 0.3
    left_expand = perspective_strength * w * 0.05

    # 垂直倾斜（X轴旋转效果）
    top_tilt = perspective_strength * h * 0.1
    bottom_tilt = perspective_strength * h * 0.03

    # 计算目标点，确保形成有效的四边形
    dst_points = np.float32([
        [max(0, 0 - left_expand), max(0, 0 + top_tilt)],                           # 左上
        [min(w-1, w-1 - right_shrink), max(0, 0 - top_tilt)],                     # 右上
        [min(w-1, w-1 - right_shrink * 0.7), min(h-1, h-1 + bottom_tilt)],       # 右下
        [max(0, 0 - left_expand * 0.3), min(h-1, h-1 - bottom_tilt)]             # 左下
    ])

    # 验证目标点形成有效四边形
    # 确保点不重合且按正确顺序排列
    for i in range(4):
        dst_points[i][0] = max(0, min(w-1, dst_points[i][0]))
        dst_points[i][1] = max(0, min(h-1, dst_points[i][1]))

    # 计算透视变换矩阵
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)

    # 应用透视变换
    result = cv2.warpPerspective(image, matrix, (w, h),
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=(255, 255, 255))

    # 添加轻微的亮度调整，模拟光照变化
    brightness_factor = 1.0 - progress * 0.1  # 侧面稍微暗一些
    result = cv2.convertScaleAbs(result, alpha=brightness_factor, beta=0)

    return result


def apply_continuous_3d_transform(image, progress):
    """
    应用连续的3D透视变换，模拟车辆从远到近的完整过程

    Args:
        image: 输入图像
        progress: 进度 (0到1)，0为远距离正面视角，1为近距离右侧面视角
    """
    h, w = image.shape[:2]

    # 1. 距离变化效果（缩放）
    # 模拟车辆从远到近，图像逐渐放大
    distance_scale = 0.7 + progress * 0.5  # 从0.7倍到1.2倍

    # 2. 视角变化效果（Y轴旋转）
    # 从正面(0°)逐渐转向右侧面(45°)
    y_rotation_angle = progress * 45  # 最大45度

    # 3. 高度变化效果（X轴轻微旋转）
    # 模拟视角高度的轻微变化
    x_rotation_angle = progress * 8  # 最大8度

    # 4. 透视强度
    perspective_strength = progress * 0.4

    # 计算变换矩阵
    # 源点（原图四个角）
    src_points = np.float32([
        [0, 0],
        [w-1, 0],
        [w-1, h-1],
        [0, h-1]
    ])

    # 目标点计算 - 模拟3D透视效果
    import math

    # Y轴旋转效果：右侧收缩，左侧保持
    right_shrink = perspective_strength * w * 0.35
    left_shift = perspective_strength * w * 0.05

    # X轴旋转效果：上下倾斜
    top_tilt = perspective_strength * h * 0.12
    bottom_tilt = perspective_strength * h * 0.04

    # 距离效果：整体缩放中心偏移
    center_x, center_y = w // 2, h // 2
    scale_offset_x = (distance_scale - 1.0) * center_x
    scale_offset_y = (distance_scale - 1.0) * center_y

    # 计算目标点
    dst_points = np.float32([
        # 左上角：轻微左移，向下倾斜，距离缩放
        [max(0, 0 - left_shift + scale_offset_x),
         max(0, 0 + top_tilt + scale_offset_y)],

        # 右上角：右侧收缩，向上倾斜，距离缩放
        [min(w-1, w-1 - right_shrink + scale_offset_x),
         max(0, 0 - top_tilt + scale_offset_y)],

        # 右下角：右侧收缩（较少），向下倾斜，距离缩放
        [min(w-1, w-1 - right_shrink * 0.6 + scale_offset_x),
         min(h-1, h-1 + bottom_tilt + scale_offset_y)],

        # 左下角：轻微左移（较少），向上倾斜，距离缩放
        [max(0, 0 - left_shift * 0.3 + scale_offset_x),
         min(h-1, h-1 - bottom_tilt + scale_offset_y)]
    ])

    # 确保目标点在有效范围内
    for i in range(4):
        dst_points[i][0] = max(-w*0.1, min(w*1.1, dst_points[i][0]))
        dst_points[i][1] = max(-h*0.1, min(h*1.1, dst_points[i][1]))

    # 应用透视变换
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    result = cv2.warpPerspective(image, matrix, (w, h),
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=(255, 255, 255))

    # 5. 光照效果
    # 模拟车辆接近时的光照变化
    brightness_factor = 0.95 + progress * 0.1  # 轻微变亮
    contrast_factor = 1.0 + progress * 0.05     # 轻微增强对比度

    result = cv2.convertScaleAbs(result, alpha=contrast_factor, beta=brightness_factor * 5)

    # 6. 轻微的模糊效果（模拟运动）
    if progress > 0.8:  # 只在接近时添加轻微模糊
        blur_strength = int((progress - 0.8) * 10)  # 0-2像素模糊
        if blur_strength > 0:
            result = cv2.GaussianBlur(result, (blur_strength*2+1, blur_strength*2+1), 0)

    return result


def generate_4th_sequence_with_interference(source_dir="3rd_1", output_dir="4nd_1"):
    """
    第四个函数：为3rd_1的图片添加Create_dataset同款干扰光

    Args:
        source_dir: 源图片目录（3rd_1）
        output_dir: 输出目录（4nd_1）
    """
    if not os.path.exists(source_dir):
        print(f"错误：找不到源目录 {source_dir}")
        print("请先运行第三个功能生成3rd_1目录")
        return

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 获取源图片列表
    source_files = [f for f in os.listdir(source_dir) if f.endswith('.jpg')]
    source_files.sort()

    if len(source_files) != 60:
        print(f"错误：源目录应包含60张图片，实际找到{len(source_files)}张")
        return

    print("开始为3rd_1图片添加Create_dataset同款干扰光...")
    print(f"源目录: {source_dir}")
    print(f"输出目录: {output_dir}")
    print("干扰光特性: 白色圆形，大小和位置连续变化")

    # 干扰光参数
    color = (255, 255, 255)  # 白色
    # 5种透明度强度，每12张图片使用一种
    alpha_levels = [0.9, 0.92, 0.94, 0.96, 1.0]

    # 定义移动轨迹的起点和终点
    # 轨迹变化幅度不大，在图像中心区域移动
    start_pos = (0.4, 0.4)   # 起始位置（相对坐标）
    end_pos = (0.6, 0.6)     # 结束位置（相对坐标）

    total_images = 0

    for i, source_file in enumerate(source_files):
        # 读取源图片
        source_path = os.path.join(source_dir, source_file)
        image = cv2.imread(source_path)

        if image is None:
            print(f"警告：无法读取图片 {source_file}")
            continue

        h, w = image.shape[:2]
        image_area = h * w

        # 计算全局进度（0到1）
        global_progress = i / 59.0

        # 计算当前组别（每12张为一组）
        group_index = i // 12  # 0, 1, 2, 3, 4
        current_alpha = alpha_levels[group_index]

        # 计算干扰光大小（从图像面积20%到80%，连续变化）
        min_area_ratio = 0.20
        max_area_ratio = 0.80
        current_area_ratio = min_area_ratio + global_progress * (max_area_ratio - min_area_ratio)
        current_area = image_area * current_area_ratio

        # 计算圆形半径（面积 = π * r²）
        import math
        radius = int(math.sqrt(current_area / math.pi))

        # 计算干扰光位置（连续变化）
        current_x_ratio = start_pos[0] + global_progress * (end_pos[0] - start_pos[0])
        current_y_ratio = start_pos[1] + global_progress * (end_pos[1] - start_pos[1])

        center_x = int(current_x_ratio * w)
        center_y = int(current_y_ratio * h)

        # 使用与Create_dataset相同的方式添加干扰光
        result = add_create_dataset_style_interference(
            image,
            center_x,
            center_y,
            radius,
            color,
            current_alpha  # 使用当前组的透明度
        )

        # 保存图片
        base_name = os.path.splitext(source_file)[0]
        output_filename = f"{base_name}_with_interference.jpg"
        output_path = os.path.join(output_dir, output_filename)
        cv2.imwrite(output_path, result)

        total_images += 1

        # 显示进度
        if (i + 1) % 12 == 0:
            group_num = (i + 1) // 12
            area_percent = current_area_ratio * 100
            print(f"  完成第{group_num}组 (第{i+1}张): 面积占比{area_percent:.1f}%, 半径{radius}像素, 透明度{current_alpha}")

    print(f"生成完成：{total_images} 张带干扰光图片保存在 {output_dir}")
    print(f"干扰光变化：面积从20%到80%，位置从({start_pos[0]:.1f},{start_pos[1]:.1f})到({end_pos[0]:.1f},{end_pos[1]:.1f})")
    print(f"透明度变化：每12张一组，强度为{alpha_levels}")
    return total_images


def add_create_dataset_style_interference(image, center_x, center_y, radius, color, alpha):
    """
    使用与Create_dataset相同的方式添加干扰光

    Args:
        image: 输入图像
        center_x, center_y: 干扰光中心位置
        radius: 干扰光半径
        color: 干扰光颜色 (R, G, B)
        alpha: 透明度
    """
    result = image.copy()
    h, w = image.shape[:2]

    # 确保圆形在图像范围内
    center_x = max(radius, min(w - radius, center_x))
    center_y = max(radius, min(h - radius, center_y))

    # 创建圆形mask
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (center_x, center_y), radius, 255, -1)

    # 使用与Create_dataset相同的光斑生成方式
    # 创建纯色光斑
    light_color = np.array([color[2], color[1], color[0]], dtype=np.uint8)  # BGR格式

    # 应用alpha混合（与add_trapezoid_laser/add_oval_laser相同的方式）
    light_area = mask > 0
    if np.sum(light_area) > 0:
        result[light_area] = (image[light_area] * (1.0 - alpha) + light_color * alpha).astype(np.uint8)

    return result


def generate_5th_sequence_with_moving_lights(image_path, output_dir="5nd_1"):
    """
    第五个函数：生成固定大小、连续移动的干扰光序列

    Args:
        image_path: 输入图像路径
        output_dir: 输出目录（默认为5nd_1）
    """
    if not os.path.exists(image_path):
        print(f"错误：找不到图像文件 {image_path}")
        return

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 读取原始图像
    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"错误：无法读取图像 {image_path}")
        return

    # 获取图像基本信息
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    h, w = original_image.shape[:2]
    image_area = h * w

    print("开始生成固定大小、连续移动的干扰光序列...")
    print(f"图像尺寸: {w}x{h}")
    print("移动模式: 前20张左→右，中20张右→左，后20张上→下")

    # 干扰光参数
    colors = [
        # (255, 0, 0),  # 红
        # (0, 255, 0),   # 绿
        # (255, 255, 255)   # 白
        (240, 240, 100)
    ]



    # 固定大小：图像面积的25%
    fixed_area_ratio = 0.25
    fixed_area = image_area * fixed_area_ratio

    # 计算圆形半径（面积 = π * r²）
    import math
    fixed_radius = int(math.sqrt(fixed_area / math.pi))

    toumingdu_list = [0.92, 0.85, 0.78, 0.7, 0.62, 0.74, 0.68, 0.6]
    total_images = 0

    for i in range(60):
        # 确定当前组别和组内位置
        group_index = i // 20  # 0, 1, 2 (每20张为一组)
        frame_in_group = i % 20  # 0-19 (组内帧号)
        group_progress = frame_in_group / 19.0  # 组内进度 (0到1)

        # 确定颜色（每20张使用同一个颜色）
        color_index = group_index  # 0, 1, 2 (每组使用不同颜色)
        current_color = colors[0]

        # 确定透明度（循环使用）

        current_alpha = toumingdu_list[group_index]

        # 根据组别确定移动模式和位置
        center_x, center_y = calculate_moving_position(
            group_index, group_progress, w, h, fixed_radius
        )

        # 使用与Create_dataset相同的方式添加干扰光
        result = add_create_dataset_style_interference(
            original_image,
            center_x,
            center_y,
            fixed_radius,
            current_color,
            current_alpha
        )

        # 保存图像
        filename = f"{base_name}_5th_moving_{i + 1:02d}.jpg"
        filepath = os.path.join(output_dir, filename)
        cv2.imwrite(filepath, result)

        total_images += 1

        # 显示进度
        if (i + 1) % 20 == 0:
            group_num = (i + 1) // 20
            movement_types = ["左→右", "右→左", "上→下"]
            color_names = ["红色", "绿色", "白色"]
            movement = movement_types[group_num - 1]
            color_name = color_names[color_index]
            print(f"  完成第{group_num}组 (第{i+1}张): {movement}移动, {color_name}, 透明度{current_alpha}")

    print(f"生成完成：{total_images} 张移动干扰光图片保存在 {output_dir}")
    print(f"干扰光特性：固定大小(半径{fixed_radius}px), 3种颜色, 3种透明度, 3种移动模式")
    return total_images


def calculate_moving_position(group_index, group_progress, width, height, radius):
    """
    计算移动干扰光的位置

    Args:
        group_index: 组索引 (0, 1, 2)
        group_progress: 组内进度 (0到1)
        width, height: 图像尺寸
        radius: 干扰光半径

    Returns:
        (center_x, center_y): 干扰光中心位置
    """
    # 确保干扰光完全在图像内的安全边距
    margin = radius + 10

    if group_index == 0:
        # 第1组：从左往右移动（降低位置）
        start_x = margin
        end_x = width - margin
        center_x = int(start_x + group_progress * (end_x - start_x))
        center_y = int(height * 0.65)  # 降低到65%位置（原来是50%）

    elif group_index == 1:
        # 第2组：从右往左移动（降低位置）
        start_x = width - margin
        end_x = margin
        center_x = int(start_x + group_progress * (end_x - start_x))
        center_y = int(height * 0.65)  # 降低到65%位置（原来是50%）

    else:  # group_index == 2
        # 第3组：从上往下移动（起始位置降低）
        center_x = width // 2  # 水平居中
        start_y = int(height * 0.3)  # 起始位置降低到30%（原来是margin）
        end_y = height - margin
        center_y = int(start_y + group_progress * (end_y - start_y))

    return center_x, center_y


def generate_6th_x_axis_rotation_sequence(image_path, output_dir="6nd_1"):
    """
    第六个函数：生成绕X轴旋转的图片序列

    Args:
        image_path: 输入图像路径
        output_dir: 输出目录（默认为6nd_1）
    """
    if not os.path.exists(image_path):
        print(f"错误：找不到图像文件 {image_path}")
        return

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 读取原始图像
    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"错误：无法读取图像 {image_path}")
        return

    # 获取图像基本信息
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    h, w = original_image.shape[:2]

    print("开始生成绕X轴旋转的图片序列...")
    print(f"图像尺寸: {w}x{h}")
    print("旋转效果: 从正面视角到俯视角度")

    # 旋转参数
    total_frames = 60
    max_rotation_angle = 60  # 最大旋转角度（度）

    print(f"旋转参数: 总帧数{total_frames}, 最大角度{max_rotation_angle}°")

    total_images = 0

    for i in range(total_frames):
        # 计算当前旋转角度
        progress = i / (total_frames - 1)  # 0到1的进度
        current_angle = progress * max_rotation_angle  # 当前旋转角度

        # 应用X轴旋转变换
        rotated_image = apply_x_axis_rotation(original_image, current_angle)

        # 保存图像
        filename = f"{base_name}_x_rotation_{i + 1:02d}.jpg"
        filepath = os.path.join(output_dir, filename)
        cv2.imwrite(filepath, rotated_image)

        total_images += 1

        # 显示进度
        if (i + 1) % 15 == 0 or i == 0 or i == total_frames - 1:
            print(f"  第{i+1:2d}张: 旋转角度{current_angle:5.1f}°")

    print(f"生成完成：{total_images} 张X轴旋转图片保存在 {output_dir}")
    print(f"旋转效果：从0°到{max_rotation_angle}°的连续X轴旋转")
    return total_images


def apply_x_axis_rotation(image, angle_degrees):
    """
    应用绕X轴的三维旋转变换（水平方向压扁效果）

    Args:
        image: 输入图像
        angle_degrees: 旋转角度（度）

    Returns:
        rotated_image: 旋转后的图像
    """
    import math

    h, w = image.shape[:2]

    # 将角度转换为弧度
    angle_rad = math.radians(angle_degrees)

    # 计算旋转后的缩放因子
    # 绕X轴旋转时，从正面看图像会在水平方向压扁
    cos_angle = math.cos(angle_rad)

    # 水平方向的缩放因子（cos值，0°时为1，90°时为0）
    horizontal_scale = abs(cos_angle)

    # 计算新的宽度（水平压扁效果）
    new_width = int(w * horizontal_scale)

    # 确保最小宽度
    if new_width < 10:
        new_width = 10

    # 先将图像在水平方向缩放
    scaled_image = cv2.resize(image, (new_width, h), interpolation=cv2.INTER_LINEAR)

    # 创建输出图像（保持原始尺寸）
    rotated_image = np.ones((h, w, 3), dtype=np.uint8) * 255  # 白色背景

    # 计算居中位置
    x_offset = (w - new_width) // 2

    # 将缩放后的图像放置在中心
    rotated_image[:, x_offset:x_offset+new_width] = scaled_image

    # 确保背景区域是纯白色
    # 不添加亮度调整，保持背景纯白

    return rotated_image

