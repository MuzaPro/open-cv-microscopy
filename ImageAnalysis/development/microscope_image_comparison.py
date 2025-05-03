import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Configuration - Edit these paths
OLD_IMAGE_PATH = r"ImageAnalysis\NIS\Msgn1_72h\exp15_metabolism_mp1_3d_day3_72hc1xy01.tif"  # Replace with actual path
NEW_IMAGE_PATH = r"ImageAnalysis\Zeiss\Msgn1_98h\exp15_Metabolism_MP1_3D_Day4_98h_b0s6c2x42275-1388y70216-1040.tif"  # Replace with actual path
OUTPUT_FOLDER = r"ImageAnalysis\development\investigation_output"  # Folder to save visualization results

def analyze_image_characteristics(old_image_path, new_image_path, output_folder):
    """Compare characteristics between old and new microscope images"""
    print(f"Analyzing images:\n- Old: {old_image_path}\n- New: {new_image_path}")
    
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")
    
    # Load images
    old_img = cv2.imread(old_image_path, cv2.IMREAD_GRAYSCALE)
    new_img = cv2.imread(new_image_path, cv2.IMREAD_GRAYSCALE)
    
    if old_img is None:
        print(f"ERROR: Could not read old image at {old_image_path}")
        return
    if new_img is None:
        print(f"ERROR: Could not read new image at {new_image_path}")
        return
    
    print(f"Successfully loaded both images")
    
    # Basic image statistics
    print("\n=== Basic Image Statistics ===")
    print(f"Old image - Min: {np.min(old_img)}, Max: {np.max(old_img)}, Mean: {np.mean(old_img):.2f}, StdDev: {np.std(old_img):.2f}")
    print(f"New image - Min: {np.min(new_img)}, Max: {np.max(new_img)}, Mean: {np.mean(new_img):.2f}, StdDev: {np.std(new_img):.2f}")
    
    # Create comparison visualizations
    # Original images
    plt.figure(figsize=(12, 10))
    plt.subplot(2, 1, 1)
    plt.title("Old Microscope Image")
    plt.imshow(old_img, cmap='gray')
    plt.axis('off')
    
    plt.subplot(2, 1, 2)
    plt.title("New Microscope Image")
    plt.imshow(new_img, cmap='gray')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "1_original_comparison.png"))
    print(f"Saved original image comparison")
    
    # Histograms
    plt.figure(figsize=(12, 6))
    plt.title("Histogram Comparison")
    old_hist = cv2.calcHist([old_img], [0], None, [256], [0, 256])
    new_hist = cv2.calcHist([new_img], [0], None, [256], [0, 256])
    
    plt.plot(old_hist, color='blue', label='Old Microscope')
    plt.plot(new_hist, color='red', label='New Microscope')
    plt.xlim([0, 256])
    plt.legend()
    plt.savefig(os.path.join(output_folder, "2_histogram_comparison.png"))
    print(f"Saved histogram comparison")
    
    # Edge detection comparison
    old_edges = cv2.Sobel(old_img, cv2.CV_64F, 1, 0, ksize=3)
    new_edges = cv2.Sobel(new_img, cv2.CV_64F, 1, 0, ksize=3)
    
    # Normalize edge images for better visualization
    old_edges = np.abs(old_edges)
    new_edges = np.abs(new_edges)
    old_edges = np.uint8(255 * old_edges / np.max(old_edges))
    new_edges = np.uint8(255 * new_edges / np.max(new_edges))
    
    plt.figure(figsize=(12, 10))
    plt.subplot(2, 1, 1)
    plt.title("Edge Detection - Old Microscope")
    plt.imshow(old_edges, cmap='gray')
    plt.axis('off')
    
    plt.subplot(2, 1, 2)
    plt.title("Edge Detection - New Microscope")
    plt.imshow(new_edges, cmap='gray')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "3_edge_detection_comparison.png"))
    print(f"Saved edge detection comparison")
    
    # Thresholding comparison
    _, old_thresh = cv2.threshold(old_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, new_thresh = cv2.threshold(new_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    plt.figure(figsize=(12, 10))
    plt.subplot(2, 1, 1)
    plt.title("Otsu Thresholding - Old Microscope")
    plt.imshow(old_thresh, cmap='gray')
    plt.axis('off')
    
    plt.subplot(2, 1, 2)
    plt.title("Otsu Thresholding - New Microscope")
    plt.imshow(new_thresh, cmap='gray')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "4_threshold_comparison.png"))
    print(f"Saved threshold comparison")
    
    # Try adaptive thresholding too
    old_adaptive = cv2.adaptiveThreshold(old_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY_INV, 21, 5)
    new_adaptive = cv2.adaptiveThreshold(new_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY_INV, 21, 5)
    
    plt.figure(figsize=(12, 10))
    plt.subplot(2, 1, 1)
    plt.title("Adaptive Thresholding - Old Microscope")
    plt.imshow(old_adaptive, cmap='gray')
    plt.axis('off')
    
    plt.subplot(2, 1, 2)
    plt.title("Adaptive Thresholding - New Microscope")
    plt.imshow(new_adaptive, cmap='gray')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "5_adaptive_threshold_comparison.png"))
    print(f"Saved adaptive threshold comparison")
    
    print("\nAnalysis complete! Review the generated images in the output folder.")
    print("Look for differences in contrast, brightness, edge clarity, and noise levels.")

if __name__ == "__main__":
    analyze_image_characteristics(OLD_IMAGE_PATH, NEW_IMAGE_PATH, OUTPUT_FOLDER)
    print("\nRun this script on multiple image pairs to identify consistent differences.")