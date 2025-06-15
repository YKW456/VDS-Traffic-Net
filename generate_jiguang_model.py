import cv2
import numpy as np
import random
import os
import shutil


def add_single_laser_dot(image_path, dot_intensity, glow_size_range, center_brightness, r, g, b):
    """
    Generate a single laser dot pattern in the center of the image with glow effect
    :param image_path: Path to input image
    :param dot_intensity: Brightness multiplier for the dot
    :param glow_size_range: Range for glow size (min, max)
    :param center_brightness: Brightness of dot center (0-255)
    :param r: Red channel value (0-255)
    :param g: Green channel value (0-255)
    :param b: Blue channel value (0-255)
    :return: Image with laser pattern added
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Unable to load image, please check path.")
        return None

    # Get image dimensions
    height, width = image.shape[:2]
    min_xy = min(image.shape[:2][0], image.shape[:2][1])

    # Create black mask image
    mask = np.zeros_like(image)

    # Randomly generate dot radius
    dot_radius = random.randint(min_xy // 4, min_xy // 2)

    dot_place_x = random.randint(image.shape[:2][0] // 5, image.shape[:2][1] // 5)
    dot_place_y = random.randint(image.shape[:2][0] // 5, image.shape[:2][1] // 5)

    # Randomly generate dot center position
    center_x = random.randint(dot_place_x, width - dot_place_x)
    center_y = random.randint(dot_place_y, height - dot_place_y)

    # Draw dot at random position
    center_color = (
    int(b * center_brightness / 255), int(g * center_brightness / 255), int(r * center_brightness / 255))
    cv2.circle(mask, (center_x, center_y), dot_radius, center_color, -1)  # Draw dot

    # Randomly generate glow size (must be odd)
    glow_size = random.randint(glow_size_range[0], glow_size_range[1])
    if glow_size % 2 == 0:  # Ensure odd number
        glow_size += 1

    # Apply Gaussian blur to mask for glow effect
    mask_blur = cv2.GaussianBlur(mask, (glow_size, glow_size), 0)

    # Blend glow effect with original image
    result = cv2.addWeighted(image, 1.0, mask_blur, dot_intensity, 0)

    return result


def add_trapezoid_laser(image_path, dot_intensity, glow_size_range, center_brightness, r, g, b, use_gaussian_blur):
    """
    Generate trapezoid laser pattern in the center of the image with glow effect
    :param image_path: Path to input image
    :param dot_intensity: Brightness multiplier for the dot
    :param glow_size_range: Range for glow size (min, max)
    :param center_brightness: Brightness of dot center (0-255)
    :param r: Red channel value (0-255)
    :param g: Green channel value (0-255)
    :param b: Blue channel value (0-255)
    :return: Image with laser pattern added
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Unable to load image, please check path.")
        return None

    # Get image dimensions
    height, width = image.shape[:2]

    # Create black mask image
    mask = np.zeros_like(image)

    # Randomly generate trapezoid parameters
    min_side = min(height, width)
    trapezoid_width = random.randint(min_side // 3, min_side // 1.2)  # Base width
    trapezoid_height = random.randint(min_side // 3, min_side // 1.2)  # Height
    trapezoid_shift = random.randint(-trapezoid_width // 4, trapezoid_width // 4)  # Tilt amount

    # Randomly generate trapezoid position
    center_x = random.randint(trapezoid_width // 2, width - trapezoid_width // 2)
    center_y = random.randint(trapezoid_height // 2, height - trapezoid_height // 2)

    # Calculate trapezoid vertices
    top_left = (center_x - trapezoid_width // 2 + trapezoid_shift, center_y - trapezoid_height // 2)
    top_right = (center_x + trapezoid_width // 2 + trapezoid_shift, center_y - trapezoid_height // 2)
    bottom_right = (center_x + trapezoid_width // 2, center_y + trapezoid_height // 2)
    bottom_left = (center_x - trapezoid_width // 2, center_y + trapezoid_height // 2)

    # Convert vertices to NumPy array
    trapezoid_points = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.int32)

    # Draw trapezoid on mask
    center_color = (
    int(b * center_brightness / 255), int(g * center_brightness / 255), int(r * center_brightness / 255))
    cv2.fillPoly(mask, [trapezoid_points], center_color)  # Draw trapezoid

    if use_gaussian_blur:
        # Randomly generate glow size (must be odd)
        glow_size = random.randint(glow_size_range[0], glow_size_range[1])
        if glow_size % 2 == 0:  # Ensure odd number
            glow_size += 1

        # Apply Gaussian blur to mask for glow effect
        mask_blur = cv2.GaussianBlur(mask, (glow_size, glow_size), 0)
    else:
        # Use original mask without blur
        mask_blur = mask

    # Blend glow effect with original image
    result = cv2.addWeighted(image, 1.0, mask_blur, dot_intensity, 0)

    return result


def add_oval_laser(image_path, dot_intensity, glow_size_range, center_brightness, r, g, b, use_gaussian_blur=True):
    """
    Generate oval laser pattern in the center of the image with glow effect
    :param image_path: Path to input image
    :param dot_intensity: Brightness multiplier for the dot
    :param glow_size_range: Range for glow size (min, max)
    :param center_brightness: Brightness of dot center (0-255)
    :param r: Red channel value (0-255)
    :param g: Green channel value (0-255)
    :param b: Blue channel value (0-255)
    :param use_gaussian_blur: Whether to use Gaussian blur (default True)
    :return: Image with laser pattern added
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Unable to load image, please check path.")
        return None

    # Get image dimensions
    height, width = image.shape[:2]

    # Create black mask image
    mask = np.zeros_like(image)

    # Randomly generate ellipse parameters
    min_side = min(height, width)
    ellipse_width = random.randint(min_side // 3, min_side // 1.2)  # Ellipse width
    ellipse_height = random.randint(min_side // 3, min_side // 1.2)  # Ellipse height
    angle = random.randint(0, 180)  # Rotation angle

    # Randomly generate ellipse position
    center_x = random.randint(ellipse_width // 2, width - ellipse_width // 2)
    center_y = random.randint(ellipse_height // 2, height - ellipse_height // 2)

    # Draw ellipse at random position
    center_color = (
    int(b * center_brightness / 255), int(g * center_brightness / 255), int(r * center_brightness / 255))
    cv2.ellipse(
        mask,
        (center_x, center_y),  # Ellipse center
        (ellipse_width // 2, ellipse_height // 2),  # Major and minor axes
        angle,  # Rotation angle
        0,  # Start angle
        360,  # End angle
        center_color,  # Color
        -1  # Fill ellipse
    )

    # If using Gaussian blur
    if use_gaussian_blur:
        # Randomly generate glow size (must be odd)
        glow_size = random.randint(glow_size_range[0], glow_size_range[1])
        if glow_size % 2 == 0:  # Ensure odd number
            glow_size += 1

        # Apply Gaussian blur to mask for glow effect
        mask_blur = cv2.GaussianBlur(mask, (glow_size, glow_size), 0)
    else:
        # Use original mask without blur
        mask_blur = mask

    # Blend glow effect with original image
    result = cv2.addWeighted(image, 1.0, mask_blur, dot_intensity, 0)

    return result


def generate(num, image_path, dot_intensity, glow_size_range, center_brightness, r, g, b, type_light, flag):
    # Create directory if it doesn't exist
    output_dir = os.path.join("jiguang", str(type_light))
    os.makedirs(output_dir, exist_ok=True)

    i = 0
    while i < num // 2:
        output_image = add_trapezoid_laser(
            image_path,
            dot_intensity=dot_intensity,
            glow_size_range=glow_size_range,
            center_brightness=center_brightness,
            r=r,
            g=g,
            b=b,
            use_gaussian_blur=flag
        )

        if output_image is not None:
            parts = image_path.split("\\")  # Split by "_"
            number = parts[1].split(".")[0]  # Extract number and remove ".jpg"
            cv2.imwrite(os.path.join(output_dir, f"{number}_{type_light}_{i}.jpg"), output_image)
            i += 1
        else:
            print("Error: Failed to generate image.")
            break

    while i < num:
        output_image = add_oval_laser(
            image_path,
            dot_intensity=dot_intensity,
            glow_size_range=glow_size_range,
            center_brightness=center_brightness,
            r=r,
            g=g,
            b=b,
            use_gaussian_blur=flag
        )

        if output_image is not None:
            parts = image_path.split("\\")  # Split by "_"
            number = parts[1].split(".")[0]  # Extract number and remove ".jpg"
            cv2.imwrite(os.path.join(output_dir, f"{number}_{type_light}_{i}.jpg"), output_image)
            i += 1
        else:
            print("Error")
            break


def generate_0(image_path, num, glow_size_range, center_brightness):
    type_light = 1
    while type_light <= 8:
        if type_light == 1:
            r, g, b = 217, 58, 54  # Dot color (yellow)
            dot_intensity = 5
            generate(num, image_path, dot_intensity, glow_size_range, center_brightness, r, g, b, type_light, flag=True)
        elif type_light == 2:
            r, g, b = 76, 179, 57  # Dot color (yellow)
            dot_intensity = 5
            generate(num, image_path, dot_intensity, glow_size_range, center_brightness, r, g, b, type_light, flag=True)
        elif type_light == 3:
            r, g, b = 36, 75, 152  # Dot color (yellow)
            dot_intensity = 5
            generate(num, image_path, dot_intensity, glow_size_range, center_brightness, r, g, b, type_light, flag=True)
        elif type_light == 4:
            r = g = b = 80
            dot_intensity = 1.7  # Glow brightness multiplier
            generate(num, image_path, dot_intensity, glow_size_range, center_brightness, r, g, b, type_light,
                     flag=False)
        elif type_light == 5:
            r = g = b = 80
            dot_intensity = 2.1  # Glow brightness multiplier
            generate(num, image_path, dot_intensity, glow_size_range, center_brightness, r, g, b, type_light,
                     flag=False)
        elif type_light == 6:
            r = g = b = 80
            dot_intensity = 2.5  # Glow brightness multiplier
            generate(num, image_path, dot_intensity, glow_size_range, center_brightness, r, g, b, type_light,
                     flag=False)
        elif type_light == 7:
            r = g = b = 80
            dot_intensity = 2.9  # Glow brightness multiplier
            generate(num, image_path, dot_intensity, glow_size_range, center_brightness, r, g, b, type_light,
                     flag=False)
        elif type_light == 8:
            r = g = b = 80
            dot_intensity = 3.5  # Glow brightness multiplier
            generate(num, image_path, dot_intensity, glow_size_range, center_brightness, r, g, b, type_light,
                     flag=False)

        type_light += 1


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