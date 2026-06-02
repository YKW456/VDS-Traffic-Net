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
    Apply continuous deformation to guarantee continuity and predictability of distortion effect

    Args:
        image: input image
        strategy: deformation mode ('perspective', 'shear', 'scale')
        deform_factor: deformation intensity ranging from 0.0 to 1.0

    Returns:
        deformed image
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
    """Continuous perspective deformation implemented via grid warping for strong continuity"""
    height, width = image.shape[:2]

    # Adopt grid distortion to generate obvious continuous warping effect
    return apply_grid_distortion(image, factor, 'perspective')

def apply_grid_distortion(image, factor, distortion_type):
    """Perform continuous smooth warping using grid-based distortion"""
    height, width = image.shape[:2]

    # Initialize mapping tables for coordinate transformation
    map_x = np.zeros((height, width), dtype=np.float32)
    map_y = np.zeros((height, width), dtype=np.float32)

    # Cubic exponential scaling to enhance continuous variation
    exponential_factor = factor * factor * factor

    for y in range(height):
        for x in range(width):
            # Normalize pixel coordinates to [0,1]
            norm_x = x / width
            norm_y = y / height

            if distortion_type == 'perspective':
                # Regular perspective warp simulating real-world distortion when vehicle approaches
                # Well-defined deformation pattern for stable continuity

                # 1. Primary deformation: trapezoidal perspective shrink (vehicle approaching simulation)
                # Top edge shrinks inward while bottom edge remains unchanged to mimic perspective projection
                perspective_strength = exponential_factor * 1.5  # Boost deformation continuity

                # Weight factor varying vertically from top to bottom
                vertical_weight = norm_y  # range 0~1: weaker influence at top, stronger at bottom

                # Trapezoidal horizontal offset: top pixels converge toward image center
                trapezoid_offset = perspective_strength * (1.0 - vertical_weight) * width * 0.3
                new_x = x + (0.5 - norm_x) * trapezoid_offset

                # 2. Auxiliary vertical compression simulating distance variation
                # More severe compression on upper region, milder compression on lower region
                vertical_compression = perspective_strength * 0.8
                compression_factor = 1.0 - vertical_compression * (1.0 - vertical_weight) * 0.4
                new_y = y * compression_factor

                # 3. Fine horizontal stretch adjustment to strengthen perspective illusion
                # Central area stretched, peripheral area compressed
                horizontal_stretch = perspective_strength * 0.6
                center_distance = abs(norm_x - 0.5)  # distance to image horizontal center
                stretch_factor = 1.0 + horizontal_stretch * (0.5 - center_distance) * vertical_weight * 0.2
                new_x = (new_x - width * 0.5) * stretch_factor + width * 0.5

                # 4. Edge smoothing to eliminate abrupt distortion at borders
                # Reduce deformation magnitude near image edges for smooth transition
                edge_distance_x = min(norm_x, 1.0 - norm_x)  # minimal distance to left/right boundary
                edge_distance_y = min(norm_y, 1.0 - norm_y)  # minimal distance to top/bottom boundary
                edge_factor = min(edge_distance_x, edge_distance_y) * 4
                edge_factor = min(1.0, edge_factor)  # clamp coefficient within [0,1]

                # Apply edge attenuation to final coordinates
                new_x = x + (new_x - x) * edge_factor
                new_y = y + (new_y - y) * edge_factor

            else:
                new_x = x
                new_y = y

            # Constrain transformed coordinates inside valid image bounds
            map_x[y, x] = np.clip(new_x, 0, width-1)
            map_y[y, x] = np.clip(new_y, 0, height-1)

    # Execute pixel remapping
    result = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    return result

def apply_barrel_distortion(image, strength):
    """Apply barrel distortion to amplify warping effect"""
    height, width = image.shape[:2]

    # Initialize coordinate lookup tables for barrel mapping
    map_x = np.zeros((height, width), dtype=np.float32)
    map_y = np.zeros((height, width), dtype=np.float32)

    center_x, center_y = width // 2, height // 2
    max_radius = min(center_x, center_y)

    for y in range(height):
        for x in range(width):
            # Calculate Euclidean distance from current pixel to image center
            dx = x - center_x
            dy = y - center_y
            radius = np.sqrt(dx*dx + dy*dy)

            if radius > 0:
                # Classic barrel distortion formula
                normalized_radius = radius / max_radius
                distorted_radius = radius * (1 + strength * normalized_radius * normalized_radius)

                # Compute updated pixel coordinates after radial warping
                scale = distorted_radius / radius
                new_x = center_x + dx * scale
                new_y = center_y + dy * scale

                map_x[y, x] = new_x
                map_y[y, x] = new_y
            else:
                map_x[y, x] = x
                map_y[y, x] = y

    # Apply coordinate remapping
    result = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    return result

def apply_continuous_shear(image, factor):
    """Continuous affine shearing transformation"""
    height, width = image.shape[:2]

    # Shear magnitude linearly correlated with input factor
    shear_x = factor * 0.8  # horizontal shear scaling coefficient
    shear_y = factor * 0.6  # vertical shear scaling coefficient

    # Construct affine shear transformation matrix
    shear_matrix = np.float32([
        [1, shear_x, 0],
        [shear_y, 1, 0]
    ])

    # Execute affine shear warp
    result = cv2.warpAffine(image, shear_matrix, (width, height),
                           borderMode=cv2.BORDER_CONSTANT,
                           borderValue=(255, 255, 255))
    return result

def apply_continuous_scale(image, factor):
    """Continuous anisotropic scaling transformation"""
    height, width = image.shape[:2]

    # Scaling ratio proportional to deformation factor
    scale_x = 1.0 + factor * 1.5
    scale_y = 1.0 + factor * 1.2

    # Build scaling matrix centered at image geometric center
    center_x, center_y = width // 2, height // 2
    scale_matrix = cv2.getRotationMatrix2D((center_x, center_y), 0, 1.0)
    scale_matrix[0, 0] = scale_x
    scale_matrix[1, 1] = scale_y

    # Perform scaling affine transformation
    result = cv2.warpAffine(image, scale_matrix, (width, height),
                           borderMode=cv2.BORDER_CONSTANT,
                           borderValue=(255, 255, 255))
    return result

def apply_random_stretch(image, stretch_range=(0.0, 0.3)):
    """
    Simulate approaching vehicle effect on traffic sign: adaptive geometric transform to minimize blank white padding

    Args:
        image: input source image
        stretch_range: lower/upper bound of stretch/compression ratio (min_ratio, max_ratio)

    Returns:
        transformed image with original resolution and intelligent background filling
    """
    if image is None:
        return None

    height, width = image.shape[:2]

    # Randomly sample deformation coefficient within predefined interval
    stretch_factor = random.uniform(stretch_range[0], stretch_range[1])

    if stretch_factor <= 0:
        return image.copy()

    # Randomly select one geometric transformation mode
    strategy = random.choice(['perspective', 'shear', 'scale_only'])

    if strategy == 'perspective':
        # Perspective projection simulating vehicle approaching from distance
        return apply_perspective_transform(image, stretch_factor)
    elif strategy == 'shear':
        # Shear transform simulating viewpoint angular shift
        return apply_shear_transform(image, stretch_factor)
    else:
        # Pure scaling simulating distance variation
        return apply_scale_transform(image, stretch_factor)

def apply_perspective_transform(image, stretch_factor):
    """Apply perspective warp with optimized smart boundary padding"""
    height, width = image.shape[:2]

    # Source quadrilateral vertices corresponding to four image corners
    src_points = np.float32([[0, 0], [width, 0], [width, height], [0, height]])

    # Random target corner offsets for realistic perspective augmentation (enhanced for training dataset diversity)
    max_offset = int(min(width, height) * stretch_factor * 0.6)

    dst_points = np.float32([
        [random.randint(0, max_offset), random.randint(0, max_offset)],
        [width - random.randint(0, max_offset), random.randint(0, max_offset)],
        [width - random.randint(0, max_offset), height - random.randint(0, max_offset)],
        [random.randint(0, max_offset), height - random.randint(0, max_offset)]
    ])

    # Calculate homography matrix for perspective projection
    perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)

    # Fill empty borders via adaptive padding algorithm
    result = apply_smart_border_fill(image, perspective_matrix, (width, height), is_perspective=True)
    return result

def apply_shear_transform(image, stretch_factor):
    """Execute affine shear transformation with optimized edge filling"""
    height, width = image.shape[:2]
    center = (width // 2, height // 2)

    # Random shear coefficients for dataset augmentation
    shear_x = stretch_factor * random.uniform(-0.4, 0.4)
    shear_y = stretch_factor * random.uniform(-0.4, 0.4)

    # Form shear affine matrix
    shear_matrix = np.float32([
        [1, shear_x, 0],
        [shear_y, 1, 0]
    ])

    # Apply smart padding on vacant border regions
    result = apply_smart_border_fill(image, shear_matrix, (width, height), is_perspective=False)
    return result

def apply_scale_transform(image, stretch_factor):
    """Conduct anisotropic scaling with intelligent edge filling"""
    height, width = image.shape[:2]
    center = (width // 2, height // 2)

    # Random scaling coefficients for data augmentation
    scale_x = 1 + stretch_factor * random.uniform(-0.5, 0.5)
    scale_y = 1 + stretch_factor * random.uniform(-0.5, 0.5)

    # Constrain scaling ratio within valid numerical range
    scale_x = max(0.6, min(1.6, scale_x))
    scale_y = max(0.6, min(1.6, scale_y))

    # Construct centered scaling affine matrix
    scale_matrix = np.float32([
        [scale_x, 0, center[0] * (1 - scale_x)],
        [0, scale_y, center[1] * (1 - scale_y)]
    ])

    # Fill blank borders adaptively
    result = apply_smart_border_fill(image, scale_matrix, (width, height), is_perspective=False)
    return result

def apply_smart_border_fill(image, transform_matrix, output_size, is_perspective=False):
    """Adaptive blank border filling using solid white background"""
    height, width = image.shape[:2]

    # Define constant white fill color for empty pixel area
    white_color = (255, 255, 255) if len(image.shape) == 3 else 255

    if is_perspective:
        # Homography-based perspective transformation
        result = cv2.warpPerspective(image, transform_matrix, output_size,
                                   borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=white_color)
    else:
        # Affine geometric transformation
        result = cv2.warpAffine(image, transform_matrix, output_size,
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=white_color)

    # Remove edge artifacts with smooth white gradient transition
    result = fix_border_artifacts(result, image, white_color)

    return result

def fix_border_artifacts(result, original, fill_color):
    """Repair edge distortion artifacts with gradual white blending"""
    height, width = result.shape[:2]

    # Generate binary mask to locate white padded pixels
    white_color = (255, 255, 255) if len(result.shape) == 3 else 255

    # Compute pixel-wise Euclidean distance to pure white value
    if len(result.shape) == 3:
        diff = np.linalg.norm(result - white_color, axis=2)
    else:
        diff = np.abs(result - white_color)

    # Create binary mask marking boundary padding zone
    border_mask = diff < 10

    # Apply smooth gradient blending on artifact regions if padding exists
    if np.any(border_mask):
        # Distance transform for smooth weight map generation
        distance = cv2.distanceTransform(border_mask.astype(np.uint8), cv2.DIST_L2, 5)
        distance = np.clip(distance / 15.0, 0, 1)  # normalize blending weight to [0,1]

        # Generate mildly blurred source image for transition mixing
        blurred = cv2.GaussianBlur(result, (9, 9), 0)

        # Blend original and blurred content only on non-white boundary area, retain pure white padding
        for c in range(result.shape[-1] if len(result.shape) == 3 else 1):
            if len(result.shape) == 3:
                mask = ~border_mask
                result[:, :, c] = np.where(mask,
                                         result[:, :, c] * (1 - distance) + blurred[:, :, c] * distance,
                                         255)
            else:
                mask = ~border_mask
                result = np.where(mask,
                                result * (1 - distance) + blurred * distance,
                                255)

    return result

# ---------- Laser Spot Generation Functions ----------
def add_trapezoid_laser(image_path, dot_intensity, glow_size_range, center_brightness, r, g, b, use_gaussian_blur, alpha=1.0, enable_vehicle_effect=True):
    """Render trapezoidal light spot with configurable transparency and approaching vehicle geometric distortion"""
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Unable to load image.")
        return None

    # Toggle vehicle approaching geometric augmentation via input flag
    if enable_vehicle_effect:
        image = apply_random_stretch(image, stretch_range=(0.1, 0.5))  # intensified deformation for training dataset
        if image is None:
            print("Error: Failed to apply vehicle approach transformation.")
            return None

    # Return raw transformed image directly when full transparency is set
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

    # Create spot mask filled with specified RGB color (OpenCV uses BGR channel order)
    mask = np.zeros_like(image, dtype=np.uint8)
    light_color = (b, g, r)
    cv2.fillPoly(mask, [pts], light_color)

    if use_gaussian_blur:
        glow_size = random.randint(glow_size_range[0], glow_size_range[1])
        glow_size += 1 if glow_size % 2 == 0 else 0
        mask = cv2.GaussianBlur(mask, (glow_size, glow_size), 0)

        # Compensate color attenuation caused by Gaussian blurring via intensity scaling
        mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        non_zero_mask = mask_gray > 0
        if np.sum(non_zero_mask) > 0:
            enhancement_factor = min(3.0, glow_size / 20.0 + 1.5)
            mask = mask.astype(np.float32)
            mask[non_zero_mask] *= enhancement_factor
            mask = np.clip(mask, 0, 255).astype(np.uint8)

    # Alpha composite only within illuminated spot region, preserve original pixels elsewhere
    result = image.copy()

    mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    light_area = mask_gray > 0

    if np.sum(light_area) > 0:
        result[light_area] = (image[light_area] * (1.0 - alpha) + mask[light_area] * alpha).astype(np.uint8)

    return result

def add_oval_laser(image_path, dot_intensity, glow_size_range, center_brightness, r, g, b, use_gaussian_blur=True, alpha=1.0, enable_vehicle_effect=True):
    """Generate elliptical luminous spot with adjustable transparency and optional vehicle approaching warp"""
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Unable to load image.")
        return None

    # Apply vehicle simulation deformation if enabled
    if enable_vehicle_effect:
        image = apply_random_stretch(image, stretch_range=(0.1, 0.5))
        if image is None:
            print("Error: Failed to apply vehicle approach transformation.")
            return None

    # Skip spot overlay when alpha equals zero
    if alpha <= 0:
        return image.copy()

    height, width = image.shape[:2]
    min_side = min(height, width)

    ellipse_width = random.randint(int(min_side // 2.5), int(min_side // 1.0))
    ellipse_height = random.randint(int(min_side // 2.5), int(min_side // 1.0))
    angle = random.randint(0, 180)
    center_x = random.randint(ellipse_width // 2, width - ellipse_width // 2)
    center_y = random.randint(ellipse_height // 2, height - ellipse_height // 2)

    # Initialize spot mask with target BGR color
    mask = np.zeros_like(image, dtype=np.uint8)
    light_color = (b, g, r)
    cv2.ellipse(mask, (center_x, center_y), (ellipse_width // 2, ellipse_height // 2), angle, 0, 360, light_color, -1)

    if use_gaussian_blur:
        glow_size = random.randint(glow_size_range[0], glow_size_range[1])
        glow_size += 1 if glow_size % 2 == 0 else 0
        mask = cv2.GaussianBlur(mask, (glow_size, glow_size), 0)

        # Boost color brightness to counteract blur-induced fading
        mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        non_zero_mask = mask_gray > 0
        if np.sum(non_zero_mask) > 0:
            enhancement_factor = min(3.0, glow_size / 20.0 + 1.5)
            mask = mask.astype(np.float32)
            mask[non_zero_mask] *= enhancement_factor
            mask = np.clip(mask, 0, 255).astype(np.uint8)

    # Perform alpha blending exclusively over luminous region
    result = image.copy()

    mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    light_area = mask_gray > 0

    if np.sum(light_area) > 0:
        result[light_area] = (image[light_area] * (1.0 - alpha) + mask[light_area] * alpha).astype(np.uint8)

    return result

def add_single_laser_dot(image_path, dot_intensity, glow_size_range, center_brightness, r, g, b, alpha=1.0):
    """Produce single circular glowing dot with configurable transparency"""
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Unable to load image.")
        return None

    # Preprocess source image with random geometric stretch
    image = apply_random_stretch(image, stretch_range=(0.1, 0.4))
    if image is None:
        print("Error: Failed to apply stretch transformation.")
        return None

    # Return raw stretched image when alpha is zero
    if alpha <= 0:
        return image.copy()

    height, width = image.shape[:2]
    min_xy = min(height, width)

    dot_radius = random.randint(int(min_xy // 3), int(min_xy // 1.5))
    dot_place_x = random.randint(width // 5, width // 5)
    dot_place_y = random.randint(height // 5, height // 5)
    center_x = random.randint(dot_place_x, width - dot_place_x)
    center_y = random.randint(dot_place_y, height - dot_place_y)

    # Generate solid-color circular spot mask
    mask = np.zeros_like(image, dtype=np.uint8)
    light_color = (b, g, r)
    cv2.circle(mask, (center_x, center_y), dot_radius, light_color, -1)

    glow_size = random.randint(glow_size_range[0], glow_size_range[1])
    glow_size += 1 if glow_size % 2 == 0 else 0
    mask = cv2.GaussianBlur(mask, (glow_size, glow_size), 0)

    # Recover color saturation lost after Gaussian smoothing
    mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    non_zero_mask = mask_gray > 0
    if np.sum(non_zero_mask) > 0:
        enhancement_factor = min(3.0, glow_size / 20.0 + 1.5)
        mask = mask.astype(np.float32)
        mask[non_zero_mask] *= enhancement_factor
        mask = np.clip(mask, 0, 255).astype(np.uint8)

    # Localized alpha blending on glowing area only
    result = image.copy()

    mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    light_area = mask_gray > 0

    if np.sum(light_area) > 0:
        result[light_area] = (image[light_area] * (1.0 - alpha) + mask[light_area] * alpha).astype(np.uint8)

    return result

# ---------- Core Dataset Generation Function ----------
def generate(num, image_path, dot_intensity, glow_size_range, center_brightness, r, g, b, type_light, flag, alpha=1.0, enable_vehicle_effect=True):
    """Generate num laser-spot images combining trapezoid & ellipse shape with built-in vehicle approaching distortion"""
    output_dir = os.path.join("jiguang", str(type_light))
    os.makedirs(output_dir, exist_ok=True)

    i = 0
    while i < num // 2:
        output_image = add_trapezoid_laser(image_path, dot_intensity, glow_size_range, center_brightness, r, g, b, flag, alpha, enable_vehicle_effect)
        if output_image is not None:
            parts = image_path.replace("\\","/").split("/")
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
    Produce multi-color multi-opacity laser spot dataset integrated with vehicle approaching geometric augmentation

    Args:
        image_path: input source image file path
        num: sample quantity generated per color configuration
        glow_size_range: lower/upper bound of luminous halo kernel size
        center_brightness: central pixel brightness of light spot
        enable_vehicle_effect: toggle vehicle-simulation geometric transform (enabled by default)
    """
    type_light = 0

    # Predefined RGB color palette list
    colors = [
        (175, 0, 175), (185, 0, 185), (195, 0, 195), (205, 0, 205), (215, 0, 215),
        (225, 0, 225), (235, 0, 235), (245, 0, 245), (255, 0, 255),
        # Purple variants
        (0, 175, 0), (0, 185, 0), (0, 195, 0), (0, 205, 0), (0, 215, 0),
        (0, 225, 0), (0, 235, 0), (0, 245, 0), (0, 255, 0),
        # Green variants
        (175, 0, 0), (185, 0, 0), (195, 0, 0), (205, 0, 0), (215, 0, 0),
        (225, 0, 0), (235, 0, 0), (245, 0, 0), (255, 0, 0)
        # Red variants
    ]
    colors_1 = [
        (240, 240, 100),  # Yellow
        (255, 255, 255)  # White
    ]

    # Opacity control lists (descending transparency values)
    dot_i_list = [1]
    dot_i_list_1 = [1.0, 0.92, 0.85, 0.78, 0.7, 0.62, 0.74, 0.68, 0.6]


    for color_idx, (r, g, b) in enumerate(colors):
        for intensity_idx, alpha in enumerate(dot_i_list):
            dot_intensity = 1.0  # spot intensity fixed, transparency fully controlled by alpha parameter
            generate(num, image_path, dot_intensity, glow_size_range, center_brightness,
                    r, g, b, type_light, flag=False, alpha=alpha,
                    enable_vehicle_effect=enable_vehicle_effect)
            type_light += 1
    # Assign unique category ID for each color-opacity combination
    for color_idx, (r, g, b) in enumerate(colors_1):
        for intensity_idx, alpha in enumerate(dot_i_list_1):
            dot_intensity = 1.0
            generate(num, image_path, dot_intensity, glow_size_range, center_brightness,
                    r, g, b, type_light, flag=False, alpha=alpha,
                    enable_vehicle_effect=enable_vehicle_effect)
            type_light += 1

    # Generate fully transparent sample (alpha=0, only vehicle warp applied without spot overlay)
    print(f"🎨 Generating fully transparent samples (type_light {type_light})...")
    generate(num, image_path, 1.0, glow_size_range, center_brightness,
            255, 255, 255, type_light, flag=False, alpha=0,
            enable_vehicle_effect=enable_vehicle_effect)

    print(f"✅ Generation complete! Total {type_light + 1} distinct categories with {num} samples per class")


def convert():
    path_jiguang = 'jiguang'
    path_test = 'classification-jiguang'
    os.makedirs(path_test, exist_ok=True)
    class_num = 14
    # Build target classification folder hierarchy
    for i in range(1, class_num + 1):
        os.makedirs(os.path.join(path_test, str(i)), exist_ok=True)

    # Recursively traverse source dataset directory
    for root, dirs, files in os.walk(path_jiguang):
        for file in files:
            if file.endswith('.jpg'):
                # Parse category index from filename string
                parts = file.split('_')
                if len(parts) >= 1 and parts[0].isdigit():
                    folder_num = int(parts[0])
                    if 1 <= folder_num <= class_num:
                        src_path = os.path.join(root, file)
                        dest_path = os.path.join(path_test, str(folder_num), file)

                        # Copy image file to target class folder
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
    Generate total 60 images (5 distinct light colors ×12 frames each; fixed halo size per group with linearly shifting spot position)
    - Source image remains unchanged without global warp; only spot mask is translated to avoid traffic sign content distortion
    - Automatically cap maximum glow size based on input image dimension to prevent full-image over-blur
    """
    # Adopt identical color palette as generate_0 function
    colors = [
        (255, 120, 100),  # Red
        (38, 219, 111),   # Green
        (80, 150, 250),   # Blue
        (240, 240, 100),  # Yellow
        (255, 255, 255)   # White
    ]

    # Select five representative opacity values consistent with generate_0's opacity list
    selected_alphas = [1.0, 0.85, 0.7, 0.6, 0.5]

    # Clamp center brightness value within valid pixel range [0,255]
    center_brightness = int(center_brightness)
    if center_brightness < 0:
        center_brightness = 0
    if center_brightness > 255:
        center_brightness = 255

    # Load source input image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load input image: {image_path}")

    h, w = image.shape[:2]
    min_side = min(h, w)
    os.makedirs(output_dir, exist_ok=True)
    img_name = os.path.splitext(os.path.basename(image_path))[0]

    num_per_light = 12  # 12 frames per light color → total 5×12 = 60 samples

    print(f"Start generating moving spot sequences: 5 groups with 12 frames per group...")

    for idx, ((r, g, b), alpha) in enumerate(zip(colors, selected_alphas)):
        color_names = ["Red", "Green", "Blue", "Yellow", "White"]
        print(f"Generating group {idx+1} ({color_names[idx]} spot, alpha={alpha})...")

        # Fix spot geometric shape unchanged throughout single group
        shape_type = random.choice(['trapezoid', 'oval'])

        # Fix Gaussian blur kernel size within specified input range for current group
        if isinstance(glow_size_range, (tuple, list)) and len(glow_size_range) == 2 and glow_size_range[0] <= glow_size_range[1]:
            glow_cand = random.randint(glow_size_range[0], glow_size_range[1])
        else:
            glow_cand = int(min_side * 0.2)

        # Upper bound restriction to avoid oversized blur kernel
        glow_upper = max(3, int(min_side * 0.25))
        glow_size = min(glow_cand, glow_upper)
        glow_size = _make_odd(glow_size)

        # Calculate spot dimension matching generate_0 sizing logic
        if shape_type == 'trapezoid':
            trapezoid_width = random.randint(int(min_side // 2.5), int(min_side // 1.0))
            trapezoid_height = random.randint(int(min_side // 2.5), int(min_side // 1.0))
            trapezoid_shift = random.randint(-trapezoid_width // 4, trapezoid_width // 4)
        else:
            ellipse_width = random.randint(int(min_side // 2.5), int(min_side // 1.0))
            ellipse_height = random.randint(int(min_side // 2.5), int(min_side // 1.0))
            angle = random.randint(0, 180)

        # Generate continuous linear moving trajectory constrained inside safe inner margin
        margin_x = max(5, w // 8)
        margin_y = max(5, h // 8)
        start_x = random.randint(margin_x, w - margin_x)
        start_y = random.randint(margin_y, h - margin_y)
        end_x = random.randint(margin_x, w - margin_x)
        end_y = random.randint(margin_y, h - margin_y)

        # Enforce adequate start-end separation for visible spot movement
        if abs(start_x - end_x) + abs(start_y - end_y) < min_side // 20:
            end_x = min(max(margin_x, end_x + (min_side // 6)), w - margin_x)
            end_y = min(max(margin_y, end_y + (min_side // 6)), h - margin_y)

        # Render 12 sequential frames with interpolated spot coordinate
        for step in range(num_per_light):
            t = step / (num_per_light - 1) if num_per_light > 1 else 0.0
            cx = int(round(start_x * (1 - t) + end_x * t))
            cy = int(round(start_y * (1 - t) + end_y * t))

            # Construct spot mask identical to training data generation pipeline
            mask = np.zeros_like(image, dtype=np.uint8)
            light_color = (b, g, r)  # OpenCV BGR channel ordering

            if shape_type == 'trapezoid':
                tl = (cx - trapezoid_width // 2 + trapezoid_shift, cy - trapezoid_height // 2)
                tr = (cx + trapezoid_width // 2 + trapezoid_shift, cy - trapezoid_height // 2)
                br = (cx + trapezoid_width // 2, cy + trapezoid_height // 2)
                bl = (cx - trapezoid_width // 2, cy + trapezoid_height // 2)
                pts = np.array([tl, tr, br, bl], dtype=np.int32)
                cv2.fillPoly(mask, [pts], light_color)
            else:
                cv2.ellipse(mask, (cx, cy), (ellipse_width // 2, ellipse_height // 2), angle, 0, 360, light_color, -1)

            # Apply Gaussian blur consistent with training dataset workflow
            mask_blur = cv2.GaussianBlur(mask, (glow_size, glow_size), 0)

            # Alpha compositing identical to training blending rule
            result = image.copy()
            mask_gray = cv2.cvtColor(mask_blur, cv2.COLOR_BGR2GRAY)
            light_area = mask_gray > 0

            if np.sum(light_area) > 0:
                result[light_area] = (image[light_area] * (1.0 - alpha) + mask_blur[light_area] * alpha).astype(np.uint8)

            out_name = f"{img_name}_moving_light{idx+1}_{step+1}.jpg"
            cv2.imwrite(os.path.join(output_dir, out_name), result)

    print(f"Generation complete: total 60 samples saved under {output_dir}")


def generate_moving_deformation_sequence_old(image_path, output_dir, glow_size_range, center_brightness=255):
    """
    Second generation function: spatially deformed image sequence matching training data distribution strictly
    - Reuse identical color palette, opacity values, dimension calculation and alpha blending rules as generate_0
    - Randomly assign single deformation mode (perspective/shear/scale) per color group
    - Smooth continuous geometric warp increment while keeping spot position fixed across group frames
    - Guarantee consistent train/test data distribution to mitigate domain shift error
    """
    # Same predefined RGB palette from generate_0
    colors = [
        (255, 120, 100),  # Red
        (38, 219, 111),   # Green
        (80, 150, 250),   # Blue
        (240, 240, 100),  # Yellow
        (255, 255, 255)   # White
    ]

    # Five selected representative opacity values consistent with generate_0 config
    selected_alphas = [1.0, 0.85, 0.7, 0.6, 0.5]

    # Sanitize input brightness value
    center_brightness = int(center_brightness)
    center_brightness = max(0, min(255, center_brightness))

    # Load source raw image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load input image: {image_path}")

    h, w = image.shape[:2]
    min_side = min(h, w)
    os.makedirs(output_dir, exist_ok=True)
    img_name = os.path.splitext(os.path.basename(image_path))[0]

    num_per_light = 12  # 12 frames per color group → total 60 images

    # Three available geometric transformation modes
    deformation_strategies = ['perspective', 'shear', 'scale']

    for idx, ((r, g, b), alpha) in enumerate(zip(colors, selected_alphas)):
        # Randomly fix deformation type for entire current group
        strategy = random.choice(deformation_strategies)

        # Lock spot center coordinate unchanged inside single group
        laser_x = random.randint(w // 4, 3 * w // 4)
        laser_y = random.randint(h // 4, 3 * h // 4)

        # Fix spot geometry and blur kernel size for current group
        shape_type = random.choice(['trapezoid', 'oval'])

        if isinstance(glow_size_range, (tuple, list)) and len(glow_size_range) == 2:
            glow_cand = random.randint(glow_size_range[0], glow_size_range[1])
        else:
            glow_cand = int(min_side * 0.2)

        glow_upper = max(3, int(min_side * 0.25))
        glow_size = min(glow_cand, glow_upper)
        glow_size = _make_odd(glow_size)

        # Spot dimension calculation aligned with generate_0 implementation
        if shape_type == 'trapezoid':
            trapezoid_width = random.randint(int(min_side // 2.5), int(min_side // 1.0))
            trapezoid_height = random.randint(int(min_side // 2.5), int(min_side // 1.0))
            trapezoid_shift = random.randint(-trapezoid_width // 4, trapezoid_width // 4)
        else:
            ellipse_width = random.randint(int(min_side // 2.5), int(min_side // 1.0))
            ellipse_height = random.randint(int(min_side // 2.5), int(min_side // 1.0))
            angle = random.randint(0, 180)

        # Define start/end deformation factor for smooth linear interpolation across frames
        if strategy == 'perspective':
            # Perspective intensity grows gradually from mild to prominent
            start_factor = 0.05
            end_factor = 0.35
        elif strategy == 'shear':
            # Shear magnitude increases from zero to obvious distortion
            start_factor = 0.0
            end_factor = 0.3
        else:  # scale
            # Scaling factor transitions from default size to apparent stretch/shrink
            start_factor = 0.05
            end_factor = 0.4

        print(f"Generating group {idx+1} ({strategy} deformation)...")

        # Render 12 frames with continuously rising deformation magnitude
        for step in range(num_per_light):
            t = step / (num_per_light - 1) if num_per_light > 1 else 0.0
            current_factor = start_factor * (1 - t) + end_factor * t

            # Apply continuous geometric deformation to base image
            deformed_image = apply_continuous_deformation(image, strategy, current_factor)

            # Generate spot mask following training dataset pipeline strictly
            mask = np.zeros_like(deformed_image, dtype=np.uint8)
            light_color = (b, g, r)

            if shape_type == 'trapezoid':
                tl = (laser_x - trapezoid_width // 2 + trapezoid_shift, laser_y - trapezoid_height // 2)
                tr = (laser_x + trapezoid_width // 2 + trapezoid_shift, laser_y - trapezoid_height // 2)
                br = (laser_x + trapezoid_width // 2, laser_y + trapezoid_height // 2)
                bl = (laser_x - trapezoid_width // 2, laser_y + trapezoid_height // 2)
                pts = np.array([tl, tr, br, bl], dtype=np.int32)
                cv2.fillPoly(mask, [pts], light_color)
            else:
                cv2.ellipse(mask, (laser_x, laser_y), (ellipse_width // 2, ellipse_height // 2), angle, 0, 360, light_color, -1)

            # Gaussian blur matching training preprocessing
            mask_blur = cv2.GaussianBlur(mask, (glow_size, glow_size), 0)

            # Standard alpha blending identical to training data creation
            result = deformed_image.copy()
            mask_gray = cv2.cvtColor(mask_blur, cv2.COLOR_BGR2GRAY)
            light_area = mask_gray > 0

            if np.sum(light_area) > 0:
                result[light_area] = (deformed_image[light_area] * (1.0 - alpha) + mask_blur[light_area] * alpha).astype(np.uint8)

            out_name = f"{img_name}_deform_{strategy}_light{idx+1}_{step+1}.jpg"
            cv2.imwrite(os.path.join(output_dir, out_name), result)

    print(f"Generation complete: total 60 deformed samples saved under {output_dir}")


def apply_strong_deformation(image, strategy, factor):
    """
    Apply intensified geometric warp exclusively for test dataset augmentation
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
    """Aggressive perspective transformation designed for test set hard samples"""
    height, width = image.shape[:2]

    # Original four corner source coordinates
    src_points = np.float32([[0, 0], [width, 0], [width, height], [0, height]])

    # Moderate yet obvious offset magnitude controlled by input factor
    max_offset = int(min(width, height) * factor * 1.0)

    dst_points = np.float32([
        [max_offset * 0.5, max_offset * 0.4],
        [width - max_offset * 0.4, max_offset * 0.5],
        [width - max_offset * 0.5, height - max_offset * 0.4],
        [max_offset * 0.4, height - max_offset * 0.5]
    ])

    # Calculate homography matrix
    perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)

    # Execute perspective warp with constant white border padding
    result = cv2.warpPerspective(image, perspective_matrix, (width, height),
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=(255, 255, 255))
    return result


def apply_strong_shear(image, factor):
    """Enhanced affine shear transformation for test dataset augmentation"""
    height, width = image.shape[:2]

    # Moderate shear coefficients tuned for visible distortion
    shear_x = factor * 0.8
    shear_y = factor * 0.6

    # Build shear affine matrix
    shear_matrix = np.float32([
        [1, shear_x, 0],
        [shear_y, 1, 0]
    ])

    # Apply shear transform with white background fill
    result = cv2.warpAffine(image, shear_matrix, (width, height),
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=(255, 255, 255))
    return result


def apply_strong_scale(image, factor):
    """Intensified anisotropic scaling transformation for test hard samples"""
    height, width = image.shape[:2]
    center = (width // 2, height // 2)

    # Tuned scaling coefficients for apparent size variation
    scale_x = 1 + factor * 0.7
    scale_y = 1 - factor * 0.5

    # Constrain scaling ratio within numerically stable bounds
    scale_x = max(0.4, min(2.0, scale_x))
    scale_y = max(0.4, min(2.0, 2.0))

    # Construct centered scaling affine matrix
    scale_matrix = np.float32([
        [scale_x, 0, center[0] * (1 - scale_x)],
        [0, scale_y, center[1] * (1 - scale_y)]
    ])

    # Execute scaling warp with constant white padding
    result = cv2.warpAffine(image, scale_matrix, (width, height),
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=(255, 255, 255))
    return result


def apply_continuous_deformation(image, strategy, factor):
    """
    Continuous smooth geometric transformation simulating real-world vehicle approaching perspective shift

    Args:
        image: input raw source image
        strategy: transformation option ('perspective', 'shear', 'scale')
        factor: deformation strength scaling factor ranging from 0.0 to 1.0

    Returns:
        geometrically distorted output image
    """
    if factor <= 0:
        return image.copy()

    height, width = image.shape[:2]

    if strategy == 'perspective':
        return apply_continuous_perspective(image, factor)
    elif strategy == 'shear':
        return apply_continuous_shear(image, factor)
    else:  # scale
        return apply_continuous_scale(image, factor)


def apply_continuous_perspective(image, factor):
    """Smooth incremental perspective projection transformation"""
    height, width = image.shape[:2]

    # Source quadrilateral vertices at four image corners
    src_points = np.float32([[0, 0], [width, 0], [width, height], [0, height]])

    # Calculate vertex offset proportional to deformation factor for continuous change
    max_offset = int(min(width, height) * factor * 0.4)

    # Fixed random seed to preserve consistent distortion direction within identical factor value
    np.random.seed(hash(str(factor)) % 2**32)

    dst_points = np.float32([
        [max_offset * 0.3, max_offset * 0.2],
        [width - max_offset * 0.2, max_offset * 0.3],
        [width - max_offset * 0.3, height - max_offset * 0.2],
        [max_offset * 0.2, height - max_offset * 0.3]
    ])

    # Compute perspective homography matrix
    perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)

    # Perform perspective warp with white constant border fill
    result = cv2.warpPerspective(image, perspective_matrix, (width, height
