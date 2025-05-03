import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

#######################################
# Change only here
main_folder_path = r'C:\Users\mrgav\Documents\GitHub\spheroid-detection\ImageAnalysis\Zeiss'
output_file_path = r'C:\Users\mrgav\Documents\GitHub\spheroid-detection\ImageAnalysis\Zeiss\zeiss_test.xlsx'  # include .xlsx

one_contour = True  # Do you only have one organoid in each image? True or False
show_images = True  # Change back to False after verification
create_graphs = False  # Do you want to create graphs? True or False

# Adjusted for high-resolution ZEN Blue images (1388x1040)
min_size = 10000  # Larger minimum size for high-res images
max_size = 2500000000000  # Maximum size to expect

pixel_to_micron = 0.645  # ZEN Blue calibration factor
identifier = 'c2'  # Put here something that will identify the phase images files

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


def clean_debris(binary_image):
    """Clean debris while preserving spheroids with sharp edges"""
    # Create kernel for morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    
    # Open operation to remove small debris (erode then dilate)
    opened = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(opened, connectivity=8)
    
    # Create output image
    clean_image = np.zeros_like(binary_image)
    
    # Filter components by size and circularity
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        
        # Create mask for this component
        component_mask = (labels == i).astype(np.uint8) * 255
        
        # Calculate circularity
        contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            perimeter = cv2.arcLength(contours[0], True)
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            
            # If area and circularity match what we expect for spheroids
            if area > min_size and circularity > 0.6:
                clean_image[labels == i] = 255
    
    return clean_image
def create_side_by_side_comparison(image_path, contour, save_dir):
    """Creates a side-by-side comparison of original and detected contour"""
    # Load original image
    original = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if original is None:
        return None
        
    # Convert to grayscale if needed
    if len(original.shape) == 3:
        original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    else:
        original_gray = original
    
    # Create BGR versions for visualization
    original_bgr = cv2.cvtColor(original_gray, cv2.COLOR_GRAY2BGR)
    contour_overlay = original_bgr.copy()
    
    # Draw contour on overlay
    cv2.drawContours(contour_overlay, [contour], -1, (0, 255, 0), 3)
    
    # Create side-by-side comparison
    h, w = original_bgr.shape[:2]
    comparison = np.zeros((h, w*2, 3), dtype=np.uint8)
    comparison[:, :w] = original_bgr
    comparison[:, w:] = contour_overlay
    
    # Add labels
    cv2.putText(comparison, "Original", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(comparison, "Detected Contour", (w+30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Save the comparison
    filename = os.path.basename(image_path)
    save_path = os.path.join(save_dir, f"comparison_{filename}")
    cv2.imwrite(save_path, comparison)
    
    return comparison

def find_largest_blob(binary_image):
    """Find the largest connected component in a binary image"""
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=8)
    
    # Find the largest component (excluding background)
    largest_size = 0
    largest_label = 0
    
    for i in range(1, num_labels):  # Start from 1 to skip background
        size = stats[i, cv2.CC_STAT_AREA]
        if size > largest_size:
            largest_size = size
            largest_label = i
    
    # Create a mask with only the largest component
    if largest_label > 0:
        largest_mask = np.zeros_like(binary_image)
        largest_mask[labels == largest_label] = 255
        return largest_mask
    
    return binary_image  # Return original if no components found


# Modify the ImageCollector class
class ImageCollector:
    def __init__(self):
        self.images = []

    def add(self, img, title):
        # Remove the filter that only adds images with "Contours:" in the title
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


def detect_spheroid_boundary(image_path):
    """Function specifically designed for detecting clear circular boundaries"""
    # Load image
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        print(f"Failed to read image: {image_path}")
        return None, None
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        print(f"Converting color image to grayscale: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Store original for visualization
    original = image.copy()
    
    # 1. Apply Canny edge detection (better for well-defined edges)
    edges = cv2.Canny(image, 50, 150)
    
    # Create a mask for visualization - use 3 channels for better visibility
    contour_image = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
    
    # Save edges visualization
    edges_vis = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    cv2.putText(edges_vis, "Detected Edges", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    image_collector.add(edges_vis, f'Canny Edges: {os.path.basename(image_path)}')
    
    # 2. Use Hough Circle detection to find the spheroid
    circles = cv2.HoughCircles(
        image, 
        cv2.HOUGH_GRADIENT, 
        dp=1, 
        minDist=image.shape[0]//2,
        param1=50, 
        param2=30, 
        minRadius=image.shape[0]//6,
        maxRadius=image.shape[0]//2
    )
    
    mask = np.zeros_like(image)
    
    # Make a copy for raw image + contour overlay
    overlay_image = contour_image.copy()
    
    if circles is not None:
        # Convert to integers
        circles = np.round(circles[0, :]).astype("int")
        
        # Take the most prominent circle
        x, y, r = circles[0]
        
        # Draw the circle with higher thickness for better visibility
        cv2.circle(contour_image, (x, y), r, (0, 255, 0), 3)  # Thicker line
        cv2.circle(contour_image, (x, y), 5, (0, 0, 255), -1)
        
        # Add annotation to know it's from Hough circles
        cv2.putText(contour_image, "Hough Circle Detection", (30, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Create mask of the circle
        cv2.circle(mask, (x, y), r, 255, -1)
        
        # Save visualization
        image_collector.add(contour_image, f'Detected Circle: {os.path.basename(image_path)}')
        
        # Create contour from circle for analysis
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create overlay to show raw image with contour
        cv2.circle(overlay_image, (x, y), r, (0, 255, 0), 3)
        cv2.putText(overlay_image, "Detected Boundary", (30, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        image_collector.add(overlay_image, f'Contours: {os.path.basename(image_path)}')
        
        return contours[0], contour_image
    
    # Fallback to contour-based approach
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by circularity
    valid_contours = []
    for contour in contours:
        if cv2.contourArea(contour) < 1000:
            continue
            
        perimeter = cv2.arcLength(contour, True)
        area = cv2.contourArea(contour)
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
        
        if circularity > 0.8:
            valid_contours.append(contour)
    
    if valid_contours:
        # Sort by area and take the largest
        largest_contour = sorted(valid_contours, key=cv2.contourArea, reverse=True)[0]
        
        # Draw contour with higher thickness
        cv2.drawContours(contour_image, [largest_contour], -1, (0, 255, 0), 3)
        
        # Add annotation about fallback method
        cv2.putText(contour_image, "Contour Detection (Fallback)", (30, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Calculate and draw centroid
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.circle(contour_image, (cX, cY), 5, (0, 0, 255), -1)
        
        # Create overlay
        cv2.drawContours(overlay_image, [largest_contour], -1, (0, 255, 0), 3)
        cv2.putText(overlay_image, "Detected Boundary", (30, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        image_collector.add(overlay_image, f'Contours: {os.path.basename(image_path)}')
        
        return largest_contour, contour_image
    
    return None, contour_image


def process_image_sobel(image_path, folder_path):
    print(f"Applying Sobel filter to: {image_path}")
    # Load the TIFF with LUT-applied RGB data
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        print(f"Failed to read image: {image_path}")
        return None
    
    # Convert RGB to grayscale properly preserving LUT-enhanced contrast
    if len(image.shape) == 3:
        print(f"Converting RGB to grayscale: {image_path}")
        # Use weighted conversion to better preserve LUT-enhanced contrast
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Display original grayscale
    if show_images:
        image_collector.add(image, f'Original: {os.path.basename(image_path)}')
    
    # Apply Sobel filter optimized for 10x objective with clear edges
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3) # Smaller kernel for sharper edges
    sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    sobel = cv2.magnitude(sobelx, sobely)
    sobel_image = cv2.normalize(sobel, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
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


def process_spheroid_image(image_path, results_df):
    print(f"Processing spheroid in: {image_path}")
    image_name = os.path.basename(image_path)
    
    # Use our specialized boundary detection
    contour, contour_image = detect_spheroid_boundary(image_path)
    
    # Check if we should create a visualization folder
    folder_path = os.path.dirname(image_path)
    images_save_dir = os.path.join(folder_path, "visualization_contours")
    
    # Create side-by-side comparison if contour is detected
    if contour is not None and os.path.exists(images_save_dir):
        comparison = create_side_by_side_comparison(image_path, contour, images_save_dir)
        if comparison is not None and show_images:
            image_collector.add(comparison, f'Comparison: {image_name}')
    
    if contour is None:
        print(f"No valid spheroid detected in {image_path}. Adding empty line.")
        empty_line = pd.DataFrame([{col: 0 for col in results_df.columns}], index=[0])
        return pd.concat([results_df, empty_line], ignore_index=True)
    
    # Calculate area and shape metrics
    area_pixels = cv2.contourArea(contour)
    area_microns = area_pixels * (pixel_to_micron ** 2)
    
    # Calculate hull area (more robust for irregular boundaries)
    hull = cv2.convexHull(contour)
    hull_area_pixels = cv2.contourArea(hull)
    hull_area_microns = hull_area_pixels * (pixel_to_micron ** 2)
    
    # Calculate other metrics
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w) / h if h > 0 else 0
    solidity = float(area_pixels) / hull_area_pixels if hull_area_pixels > 0 else 0
    
    # Display detected contour
    if show_images:
        image_collector.add(contour_image, f'Contours: {image_name}')
    
    # Measure intensity in other channels
    intensity_data = {}
    for suffix in ["c0", "c1"]:
        fluorescent_image_path = image_path.replace("c2", suffix)
        if not os.path.exists(fluorescent_image_path):
            print(f"Fluorescent image not found: {fluorescent_image_path}")
            continue

        fluorescent_image = cv2.imread(fluorescent_image_path, cv2.IMREAD_UNCHANGED)
        if fluorescent_image is None:
            print(f"Failed to read fluorescent image: {fluorescent_image_path}")
            continue
            
        # Convert to grayscale if needed
        if len(fluorescent_image.shape) == 3:
            print(f"Converting color fluorescent image to grayscale: {fluorescent_image_path}")
            fluorescent_image = cv2.cvtColor(fluorescent_image, cv2.COLOR_BGR2GRAY)

        intensity = measure_fluorescence_intensity(fluorescent_image, contour)
        intensity_data[f'MeanIntensity{suffix}'] = intensity
    
    # Add to results
    new_row = {
        'Area': hull_area_microns,
        'Area_Pixels': area_pixels,
        'Aspect_Ratio': aspect_ratio,
        'Solidity': solidity,
        **intensity_data
    }
    
    print(f"Adding row with Area_Pixels: {area_pixels}, Area_Microns: {hull_area_microns}")
    return pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)


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
    import shutil

    results_df = pd.DataFrame(columns=['Area', 'Area_Pixels', 'Aspect_Ratio', 'Solidity', 'MeanIntensityc0', 'MeanIntensityc1'])
    
    # Create a special visualization directory
    folder_name = os.path.basename(folder_path)
    images_save_dir = os.path.join(folder_path, f"visualization_contours")
    
    # Handle existing directory
    if os.path.exists(images_save_dir):
        try:
            shutil.rmtree(images_save_dir)
            print(f"Removed existing directory: {images_save_dir}")
        except Exception as e:
            print(f"Could not remove existing directory: {e}")
            import time
            timestamp = int(time.time())
            images_save_dir = os.path.join(folder_path, f"visualization_contours_{timestamp}")
    
    try:
        os.makedirs(images_save_dir)
        print(f"Created directory: {images_save_dir}")
    except Exception as e:
        print(f"Warning: Could not create directory {images_save_dir}: {e}")
        images_save_dir = None
    
    # Find all phase contrast images
    image_files = [f for f in os.listdir(folder_path) if
                  identifier in f and f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
    print(f"Found {len(image_files)} images to process in {folder_path}")
    
    for filename in image_files:
        image_path = os.path.join(folder_path, filename)
        
        # Use our specialized spheroid detection function
        results_df = process_spheroid_image(image_path, results_df)
        
        # Display and save collected images after processing each file
        if show_images and images_save_dir:
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