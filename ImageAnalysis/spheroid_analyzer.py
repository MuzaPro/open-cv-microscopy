import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

#######################################
# Change only here
main_folder_path = r'C:\Users\mrgav\Desktop\Lara\ImageAnalysis\NIS'
output_file_path = r'C:\Users\mrgav\Desktop\Lara\ImageAnalysis\NIS\Msgn1_test.xlsx'  # include .xlsx

one_contour = True  # Do you only have one organoid in each image? True or False
show_images = True  # Do you want to see the images as the code runs? True or False
create_graphs = True  # Do you want to create graphs? True or False

min_size = 10000  # What is the minimum size you expect?
max_size = 25000000000  # What is the maximum size you expect?

pixel_to_micron = 0.77
identifier = 'c1'  # Put here something that will identify the phase images files

t_test = False  # Do you want to conduct a t-test? True or False


########################################

# Create a class to store and display images

# Add import for ensuring directories exist

# Replace skimage functions with equivalent cv2 functions
def threshold_otsu(image):
    """OpenCV implementation of Otsu thresholding"""
    # Ensure image is 8-bit
    if image.dtype != np.uint8:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    ret, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return ret  # Return the threshold value


def binary_fill_holes(binary_image):
    """OpenCV implementation of binary_fill_holes"""
    # Ensure binary image is 8-bit
    if binary_image.dtype != np.uint8:
        binary_image = binary_image.astype(np.uint8) * 255

    # Find contours
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create mask and fill contours
    mask = np.zeros_like(binary_image)
    for contour in contours:
        cv2.drawContours(mask, [contour], 0, 255, -1)

    return mask


def binary_closing(binary_image, kernel):
    """OpenCV implementation of binary_closing"""
    # Ensure binary image is 8-bit
    if binary_image.dtype != np.uint8:
        binary_image = binary_image.astype(np.uint8) * 255

    # Apply morphological closing
    closed = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)

    return closed


def remove_small_objects(binary_image, min_size):
    """OpenCV implementation of remove_small_objects"""
    # Ensure binary image is 8-bit
    if binary_image.dtype != np.uint8:
        binary_image = binary_image.astype(np.uint8) * 255

    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=8)

    # Create output image
    output = np.zeros_like(binary_image)

    # Start from 1 to skip background
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            output[labels == i] = 255

    return output


# Modify the ImageCollector class
class ImageCollector:
    def __init__(self):
        self.images = []

    def add(self, img, title):
        # Only add images with "Contours:" in the title
        if "Contours:" in title:
            self.images.append((img.copy(), title))

    def display_all(self, save_path=None):
        n = len(self.images)
        if n == 0:
            return

        # Calculate grid size
        cols = min(3, n)
        rows = (n + cols - 1) // cols

        plt.figure(figsize=(15, 5 * rows))
        for i, (img, title) in enumerate(self.images):
            plt.subplot(rows, cols, i + 1)

            # Convert BGR to RGB if image has 3 channels
            if len(img.shape) == 3 and img.shape[2] == 3:
                plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            else:
                plt.imshow(img, cmap='gray')

            plt.title(title)
            plt.axis('off')

            # Save individual image if save_path is provided
            if save_path:
                # Create sanitized filename from title
                filename = title.replace(":", "_").replace(" ", "_")
                img_save_path = os.path.join(save_path, f"{filename}.png")

                # Save the image
                if len(img.shape) == 3 and img.shape[2] == 3:
                    cv2.imwrite(img_save_path, img)
                else:
                    cv2.imwrite(img_save_path, img)

        # Save the figure if save_path is provided
        if save_path:
            plt_save_path = os.path.join(save_path, "summary_figure.png")
            plt.tight_layout()
            plt.savefig(plt_save_path)

        if show_images:
            plt.tight_layout()
            plt.show()

        self.images = []  # Clear images after displaying/saving


# Create global image collector
image_collector = ImageCollector()


def measure_fluorescence_intensity(image, contour):
    mask = np.zeros_like(image, dtype="uint8")
    cv2.drawContours(mask, [contour], -1, 255, -1)
    mean_intensity = cv2.mean(image, mask=mask)[0]
    return mean_intensity


def process_image_sobel(image_path, folder_path):
    print(f"Applying Sobel filter to: {image_path}")
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        print(f"Failed to read image: {image_path}")
        return None

    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    sobel = cv2.magnitude(sobelx, sobely)
    sobel_image = np.uint8(sobel)

    if show_images:
        image_collector.add(sobel_image, f'Sobel Filter: {os.path.basename(image_path)}')

    return sobel_image


def remove_noise(binary_image, min_size):
    labels_count, labeled_image, stats, centroids = cv2.connectedComponentsWithStats(
        binary_image, 8, cv2.CV_32S)
    new_image = np.zeros_like(binary_image)
    for i in range(1, labels_count):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > min_size:
            new_image[labeled_image == i] = 255
    return new_image


def debug_contour_areas(contours, pixel_to_micron, image_name):
    """Print detailed area information for all contours"""
    print(f"\n=== DEBUG AREAS for {image_name} ===")
    print(f"Pixel to micron ratio: {pixel_to_micron}")
    print(f"Number of contours: {len(contours)}")

    for i, cnt in enumerate(contours[:5]):  # Show first 5 contours
        # Area in pixels
        area_pixels = cv2.contourArea(cnt)

        # Area in square microns
        area_microns = area_pixels * (pixel_to_micron ** 2)

        # Perimeter in pixels
        perimeter_pixels = cv2.arcLength(cnt, True)

        # Convex hull area in pixels
        hull = cv2.convexHull(cnt)
        hull_area_pixels = cv2.contourArea(hull)

        # Convex hull area in square microns
        hull_area_microns = hull_area_pixels * (pixel_to_micron ** 2)

        print(f"Contour {i}:")
        print(f"  Raw area (pixels²): {area_pixels:.2f}")
        print(f"  Raw area (microns²): {area_microns:.2f}")
        print(f"  Hull area (pixels²): {hull_area_pixels:.2f}")
        print(f"  Hull area (microns²): {hull_area_microns:.2f}")
        print(f"  Perimeter (pixels): {perimeter_pixels:.2f}")

    print("===========================\n")

def process_contours(image, image_path, results_df, min_area_um2, max_area_um2, pixel_to_micron=pixel_to_micron):
    print(f"Processing contours in: {image_path}")
    image_name = os.path.basename(image_path)
    added_valid_contour = False  # Track if we added a valid contour

    # Apply stronger Gaussian blur to smooth noise
    blur = cv2.GaussianBlur(image, (7, 7), 3)  # Increased kernel size and sigma
    if show_images:
        image_collector.add(blur, f'Gaussian Blur: {image_name}')

    # Use adaptive thresholding instead of fixed threshold
    # This helps with varying brightness levels across images
    mask = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, 21, 5)

    # Alternative: you can still use cv2.inRange but with optimized values
    # ret, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if show_images:
        image_collector.add(mask, f'Mask: {image_name}')

    # Add morphological operations to smooth contours
    # Create kernel for morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # Close small holes inside the foreground objects
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Open to remove small noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Dilate to expand boundaries slightly
    mask = cv2.dilate(mask, kernel, iterations=1)

    if show_images:
        image_collector.add(mask, f'Morphed Mask: {image_name}')

    masked_img = cv2.bitwise_and(image, image, mask=mask)
    # Convert to single channel if needed
    if len(masked_img.shape) > 2:
        masked_img = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY)

    if show_images:
        image_collector.add(masked_img, f'Masked Image: {image_name}')

    rem_noise = remove_noise(masked_img, min_area_um2)
    if show_images:
        image_collector.add(rem_noise, f'Removed Noise: {image_name}')

    # Apply one more round of smoothing specifically to improve circularity
    smooth_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    smooth_contours = cv2.morphologyEx(rem_noise, cv2.MORPH_CLOSE, smooth_kernel, iterations=3)

    if show_images:
        image_collector.add(smooth_contours, f'Smoothed for Circularity: {image_name}')

    contours, _ = cv2.findContours(smooth_contours, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    debug_contour_areas(contours, pixel_to_micron, image_name)

    print(f"Debug for {image_name}: Found {len(contours)} initial contours")

    suitable_contours = [cnt for cnt in contours if
                         min_area_um2 < cv2.contourArea(cv2.convexHull(cnt)) * (pixel_to_micron ** 2) < max_area_um2]

    print(f"After size filtering: {len(suitable_contours)} contours remain")

    if not suitable_contours:
        if one_contour:
            print(f"No suitable contours found in {image_path}. Adding empty line.")
            empty_line = pd.DataFrame([{col: 0 for col in results_df.columns}], index=[0])
            results_df = pd.concat([results_df, empty_line], ignore_index=True)
            print(f"After adding empty line: {results_df}")
            return results_df
        else:
            return results_df

    if one_contour:
        suitable_contours = sorted(suitable_contours, key=lambda c: cv2.contourArea(c), reverse=True)
        print(f"Sorted contours by size, largest first")

    contour_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Process all contours and collect valid ones
    valid_contours = []
    valid_contour_data = []

    for n, contour in enumerate(suitable_contours):
        # Apply contour approximation to further smooth the contour
        epsilon = 0.005 * cv2.arcLength(contour, True)
        approx_contour = cv2.approxPolyDP(contour, epsilon, True)

        # Calculate circularity with the smoothed contour
        perimeter = cv2.arcLength(approx_contour, True)
        area = cv2.contourArea(approx_contour)
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0

        print(f"  Contour {n}: Area={area * (pixel_to_micron ** 2):.2f}μm², Circularity={circularity:.2f}")

        # Skip non-circular contours
        if circularity < 0.5:
            print(f"    Rejected: circularity {circularity:.2f} < 0.5")
            continue

        # Skip contours that are cut off by the edge
        x, y, w, h = cv2.boundingRect(approx_contour)
        #if x == 0 or y == 0 or x + w == image.shape[1] or y + h == image.shape[0]:
         #   print(f"    Rejected: contour cut off by edge (x={x}, y={y}, w={w}, h={h})")
          #  continue

        hull = cv2.convexHull(approx_contour)
        hull_area = cv2.contourArea(hull) * (pixel_to_micron ** 2)
        solidity = float(area) / (cv2.contourArea(hull)) if cv2.contourArea(hull) > 0 else 0

        print(f"    Solidity={solidity:.2f}")

        if solidity < 0.4 or solidity > 1.8:
            print(f"    Rejected: solidity {solidity:.2f} outside range [0.4, 1.8]")
            continue

        aspect_ratio = float(w) / h
        print(f"    Aspect ratio={aspect_ratio:.2f}")

        if aspect_ratio > 4 or aspect_ratio < 1 / 4:
            print(f"    Rejected: aspect ratio {aspect_ratio:.2f} outside range [0.25, 4]")
            continue

        print(f"    VALID: This contour passed all criteria!")

        # Store the valid contour and its data
        valid_contours.append(approx_contour)
        valid_contour_data.append({
            'contour': approx_contour,
            'area': area,
            'hull_area': hull_area,
            'aspect_ratio': aspect_ratio,
            'solidity': solidity,
            'x': x, 'y': y, 'w': w, 'h': h
        })

    # Now process the valid contours
    if one_contour and valid_contours:
        # Sort valid contours by area and pick the largest
        valid_contour_data.sort(key=lambda x: x['area'], reverse=True)
        print(
            f"Found {len(valid_contours)} valid contours, selecting the largest one with area {valid_contour_data[0]['area'] * (pixel_to_micron ** 2):.2f}μm²")

        # Process only the largest valid contour
        contour_data = valid_contour_data[0]
        contour = contour_data['contour']

        M = cv2.moments(contour)
        cX = int(M["m10"] / M["m00"]) if M["m00"] != 0 else 0
        cY = int(M["m01"] / M["m00"]) if M["m00"] != 0 else 0

        cv2.drawContours(contour_image, [contour], -1, (0, 255, 0), 2)
        cv2.circle(contour_image, (cX, cY), 5, (255, 0, 0), -1)

        if show_images:
            image_collector.add(contour_image, f'Contours: {image_name}')

        intensity_data = {}
        for suffix in ["c2", "c3"]:
            fluorescent_image_path = image_path.replace("c1", suffix)
            if not os.path.exists(fluorescent_image_path):
                print(f"Fluorescent image not found: {fluorescent_image_path}")
                continue

            fluorescent_image = cv2.imread(fluorescent_image_path, cv2.IMREAD_GRAYSCALE)
            if fluorescent_image is None:
                print(f"Failed to read fluorescent image: {fluorescent_image_path}")
                continue

            intensity = measure_fluorescence_intensity(fluorescent_image, contour)
            intensity_data[f'MeanIntensity{suffix}'] = intensity

        # In the one_contour section:
        new_row = {
            'Area': contour_data['hull_area'],
            'Area_Pixels': contour_data['area'],  # Add this line
            'Aspect_Ratio': contour_data['aspect_ratio'],
            'Solidity': contour_data['solidity'],
            **intensity_data
        }
        print(f"Adding row with Area_Pixels: {contour_data['area']}, Area_Microns: {contour_data['hull_area']}")
        results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)
        added_valid_contour = True

    elif not one_contour:
        # Process all valid contours (if not in one_contour mode)
        for contour_data in valid_contour_data:
            contour = contour_data['contour']

            M = cv2.moments(contour)
            cX = int(M["m10"] / M["m00"]) if M["m00"] != 0 else 0
            cY = int(M["m01"] / M["m00"]) if M["m00"] != 0 else 0

            cv2.drawContours(contour_image, [contour], -1, (0, 255, 0), 2)
            cv2.circle(contour_image, (cX, cY), 5, (255, 0, 0), -1)

            if show_images:
                image_collector.add(contour_image, f'Contours: {image_name}')

            intensity_data = {}
            for suffix in ["c2", "c3"]:
                fluorescent_image_path = image_path.replace("c1", suffix)
                if not os.path.exists(fluorescent_image_path):
                    print(f"Fluorescent image not found: {fluorescent_image_path}")
                    continue

                fluorescent_image = cv2.imread(fluorescent_image_path, cv2.IMREAD_GRAYSCALE)
                if fluorescent_image is None:
                    print(f"Failed to read fluorescent image: {fluorescent_image_path}")
                    continue

                intensity = measure_fluorescence_intensity(fluorescent_image, contour)
                intensity_data[f'MeanIntensity{suffix}'] = intensity

            new_row = {
                'Area': contour_data['hull_area'],
                'Aspect_Ratio': contour_data['aspect_ratio'],
                'Solidity': contour_data['solidity'],
                **intensity_data
            }
            results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)
            added_valid_contour = True

    # If we're looking for one contour but didn't add any valid ones
    if one_contour and not added_valid_contour:
        print(f"No valid contours processed in {image_path}. Adding empty line.")
        empty_line = pd.DataFrame([{col: 0 for col in results_df.columns}], index=[0])
        results_df = pd.concat([results_df, empty_line], ignore_index=True)

    return results_df

def pval_ttest(filepath, control, variances, outputpath):
    # Load the Excel file
    file_path = filepath
    xls = pd.ExcelFile(file_path)

    # Create a DataFrame to store p-values
    params = []
    sheet_names = []
    p_two_tails = []
    p_one_tails = []

    # Assuming 'DMSO' is the control group and its data is in a separate sheet
    control_data = xls.parse(control)

    # Iterate over each sheet in the Excel file
    for sheet_name in xls.sheet_names:
        if sheet_name == control:  # Skip the control sheet
            continue

        # Reading the sheet into a DataFrame
        sheet_data = xls.parse(sheet_name)
        # Iterate over each column (parameter) in the sheet
        for param in sheet_data.columns:
            if param == 'Condition':  # Skip non-data columns
                continue

            # Prepare data for t-test
            control_values = control_data[param].dropna()
            test_values = sheet_data[param].dropna()

            # Perform two-tailed t-test
            t_stat, p_two_tail = stats.ttest_ind(control_values, test_values, equal_var=variances,
                                                 nan_policy='omit')

            # Perform one-tailed t-test
            p_one_tail = p_two_tail / 2 if t_stat > 0 else 1 - (p_two_tail / 2)

            params.append(param)
            sheet_names.append(sheet_name)
            p_two_tails.append(p_two_tail)
            p_one_tails.append(p_one_tail)

    # Add p-values to the DataFrame
    p_values = pd.DataFrame({
        'Parameter': params,
        'Condition': sheet_names,
        'P_Value_Two_Tail': p_two_tails,
        'P_Value_One_Tail': p_one_tails
    })

    # Save the p-values to a new Excel file
    output_file_path = outputpath
    p_values.to_excel(output_file_path, index=False)


def ensure_dir(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)


def process_folder(folder_path, min_area_um2, max_area_um2):
    import shutil  # Make sure this import is at the top of your file

    results_df = pd.DataFrame(columns=['Area', 'Aspect_Ratio', 'Solidity', 'MeanIntensityc2', 'MeanIntensityc3'])

    # Create a subdirectory for saving images
    folder_name = os.path.basename(folder_path)
    time_suffix = folder_name.split('_')[1] if '_' in folder_name else 'unknown'
    images_save_dir = os.path.join(folder_path, f"processed_images_{time_suffix}")

    # Check if directory exists and handle it properly
    if os.path.exists(images_save_dir):
        try:
            # Try to remove the folder and its contents
            shutil.rmtree(images_save_dir)
            print(f"Removed existing directory: {images_save_dir}")
        except Exception as e:
            # If removal fails, use a different name
            print(f"Could not remove existing directory: {e}")
            # Add timestamp to make unique name
            import time
            timestamp = int(time.time())
            images_save_dir = os.path.join(folder_path, f"processed_images_{time_suffix}_{timestamp}")
            print(f"Using alternative directory: {images_save_dir}")

    # Now create the directory
    try:
        os.makedirs(images_save_dir)
        print(f"Created directory: {images_save_dir}")
    except Exception as e:
        print(f"Warning: Could not create directory {images_save_dir}: {e}")
        # Continue without saving images
        images_save_dir = None

    # Count the number of images to process
    image_files = [f for f in os.listdir(folder_path) if
                   identifier in f and f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
    print(f"Found {len(image_files)} images to process in {folder_path}")

    for filename in image_files:
        image_path = os.path.join(folder_path, filename)
        sobel_image = process_image_sobel(image_path, folder_path)
        if sobel_image is not None:
            results_df = process_contours(sobel_image, image_path, results_df, min_area_um2, max_area_um2)

        # Display and save collected images after processing each file
        if show_images and images_save_dir:  # Only try to save if directory exists
            image_collector.display_all(save_path=images_save_dir)

    return results_df


def get_folders(main_folder_path):
    """Get a list of all subdirectories in the main folder."""
    return [os.path.join(main_folder_path, d) for d in os.listdir(main_folder_path) if
            os.path.isdir(os.path.join(main_folder_path, d))]


import re


def get_time_suffix(folder_name):
    match = re.search(r'_(-?\d+)$', folder_name)  # Added -? to match optional minus sign
    return int(match.group(1)) if match else float('inf')  # Use infinity for sorting


def process_multiple_folders(main_folder_path, min_area_um2, max_area_um2, output_file_path):
    folder_paths = get_folders(main_folder_path)
    writer = pd.ExcelWriter(output_file_path, engine='openpyxl')
    results_by_condition = {}

    folder_paths = get_folders(main_folder_path)
    folder_paths = sorted(folder_paths, key=lambda x: get_time_suffix(os.path.basename(x)))

    # Count total folders for processing
    total_folders = len(folder_paths)
    print(f"Found {total_folders} folders to process.")

    for i, folder_path in enumerate(folder_paths):
        folder_name = os.path.basename(folder_path)
        parts = folder_name.split('_')

        if len(parts) >= 2:
            condition = parts[0]
            time_suffix = parts[1]
        else:
            # Handle folders without the expected format
            condition = folder_name
            time_suffix = "0"  # Default to time 0
        print(f"Processing folder {i + 1}/{total_folders}: {folder_path} for condition: {condition}")

        results_df = process_folder(folder_path, min_area_um2, max_area_um2)

        # Print a summary of the results for this folder
        print(f"  Processed folder: {os.path.basename(folder_path)}")
        print(f"  Found {len(results_df)} results")

        results_df = results_df.rename(columns={col: f"{time_suffix}_{col}" for col in results_df.columns})
        if condition in results_by_condition:
            results_by_condition[condition] = pd.concat([results_by_condition[condition], results_df], axis=1)
        else:
            results_by_condition[condition] = results_df

    # Write each condition's results to a separate tab
    for condition, df in results_by_condition.items():
        df.to_excel(writer, sheet_name=condition, index=False)

    writer.close()
    print(f"Results saved to {output_file_path}")


if t_test:
    # Get the required inputs for the pval_ttest function
    control_tab = input("What is the name of the control tab? ")
    assume_equal_variances = input("Do you assume equal variances? True or False: ") == 'True'
    p_values_output_path = input(
        "What is the path to the output file for p-values? Include the name of the file and the ending .xlsx: ")

    # Run the pval_ttest function with the provided inputs
    pval_ttest(output_file_path, control_tab, assume_equal_variances, p_values_output_path)

    print(f"P-values have been saved to {p_values_output_path}")

process_multiple_folders(main_folder_path, min_size, max_size, output_file_path)