import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Configuration - Edit these paths
IMAGE_PATH = r"ImageAnalysis\Zeiss\Msgn1_98h\exp15_Metabolism_MP1_3D_Day4_98h_b0s6c2x42275-1388y70216-1040.tif" # Replace with actual path
OUTPUT_FOLDER = r"ImageAnalysis\development\investigation_output\pipeline_debug"  # Folder to save debug results
PIXEL_TO_MICRON = 0.77  # Match your main script's value
MIN_SIZE = 100000  # Match your main script's value
MAX_SIZE = 25000000000  # Match your main script's value

def debug_pipeline(image_path, output_folder, pixel_to_micron, min_size, max_size):
    """Debug each stage of the image processing pipeline"""
    print(f"Debugging pipeline for: {image_path}")
    
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Load the image
    original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if original is None:
        print(f"ERROR: Could not read image at {image_path}")
        return
    
    # Stage 1: Original image
    stages = {"1_Original": original.copy()}
    
    # Stage 2: Sobel edge detection
    sobelx = cv2.Sobel(original, cv2.CV_64F, 1, 0, ksize=7)
    sobely = cv2.Sobel(original, cv2.CV_64F, 0, 1, ksize=7)
    sobel = cv2.magnitude(sobelx, sobely)
    sobel = np.uint8(255 * sobel / np.max(sobel))
    stages["2_Sobel"] = sobel.copy()
    
    # Stage 3: Gaussian blur
    blur = cv2.GaussianBlur(sobel, (7, 7), 3)
    stages["3_Blur"] = blur.copy()
    
    # Stage 4: Adaptive thresholding
    mask = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 3, 1)
    stages["4_Threshold"] = mask.copy()
    
    # Stage 5: Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)
    dilated = cv2.dilate(opened, kernel, iterations=1)
    stages["5_Morphology"] = dilated.copy()
    
    # Stage 6: Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_img = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
    stages["6_All_Contours"] = contour_img.copy()
    
    print(f"Found {len(contours)} initial contours")
    
    # Stage 7: Filter by size
    size_filtered = []
    for cnt in contours:
        area = cv2.contourArea(cv2.convexHull(cnt)) * (pixel_to_micron ** 2)
        if min_size < area < max_size:
            size_filtered.append(cnt)
    
    size_filtered_img = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(size_filtered_img, size_filtered, -1, (0, 255, 0), 2)
    stages["7_Size_Filtered"] = size_filtered_img
    
    print(f"After size filtering: {len(size_filtered)} contours remain")
    
    # Stage 8: Shape filtering
    shape_filtered = []
    for cnt in size_filtered:
        # Smooth contour
        epsilon = 0.005 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        
        # Calculate metrics
        area = cv2.contourArea(approx)
        perimeter = cv2.arcLength(approx, True)
        circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
        
        hull = cv2.convexHull(approx)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area if hull_area > 0 else 0
        
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = float(w) / h if h > 0 else 0
        
        # Check criteria
        if circularity >= 0.5 and 0.4 <= solidity <= 1.8 and 0.25 <= aspect_ratio <= 4:
            shape_filtered.append(approx)
            
            # Print details of passing contours
            print(f"\nPASSING CONTOUR:")
            print(f"  Area: {area * (pixel_to_micron ** 2):.2f} µm²")
            print(f"  Circularity: {circularity:.2f}")
            print(f"  Solidity: {solidity:.2f}")
            print(f"  Aspect ratio: {aspect_ratio:.2f}")
        else:
            # Print details of failing contours
            print(f"\nFAILING CONTOUR:")
            print(f"  Area: {area * (pixel_to_micron ** 2):.2f} µm²")
            print(f"  Circularity: {circularity:.2f} {'PASS' if circularity >= 0.5 else 'FAIL'}")
            print(f"  Solidity: {solidity:.2f} {'PASS' if 0.4 <= solidity <= 1.8 else 'FAIL'}")
            print(f"  Aspect ratio: {aspect_ratio:.2f} {'PASS' if 0.25 <= aspect_ratio <= 4 else 'FAIL'}")
    
    shape_filtered_img = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(shape_filtered_img, shape_filtered, -1, (0, 255, 0), 2)
    stages["8_Shape_Filtered"] = shape_filtered_img
    
    print(f"After shape filtering: {len(shape_filtered)} contours remain")
    
    # Save all stages
    rows = (len(stages) + 1) // 2
    plt.figure(figsize=(15, rows * 5))
    
    for i, (name, img) in enumerate(stages.items()):
        plt.subplot(rows, 2, i + 1)
        
        if len(img.shape) == 3:
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(img, cmap='gray')
            
        plt.title(name)
        plt.axis('off')
        
        # Also save individual images
        if len(img.shape) == 3:
            cv2.imwrite(os.path.join(output_folder, f"{name}.png"), img)
        else:
            cv2.imwrite(os.path.join(output_folder, f"{name}.png"), img)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "pipeline_summary.png"))
    print(f"Saved pipeline debug results to {output_folder}")
    
    # Return key insights
    return {
        "total_contours": len(contours),
        "size_filtered_contours": len(size_filtered),
        "final_valid_contours": len(shape_filtered)
    }

if __name__ == "__main__":
    results = debug_pipeline(IMAGE_PATH, OUTPUT_FOLDER, PIXEL_TO_MICRON, MIN_SIZE, MAX_SIZE)
    print("\nPipeline Debug Summary:")
    print(f"- Initial contours: {results['total_contours']}")
    print(f"- After size filter: {results['size_filtered_contours']}")
    print(f"- Final valid contours: {results['final_valid_contours']}")
    
    if results['final_valid_contours'] == 0:
        if results['size_filtered_contours'] > 0:
            print("\nProblem identified: Contours are being rejected by shape criteria")
            print("Check the output for which specific criteria (circularity, solidity, aspect ratio) are failing")
        elif results['total_contours'] > 0:
            print("\nProblem identified: Contours are being rejected by size criteria")
            print(f"Consider adjusting MIN_SIZE ({MIN_SIZE}) and MAX_SIZE ({MAX_SIZE})")
        else:
            print("\nProblem identified: No contours are being detected at all")
            print("Check the thresholding and edge detection steps")