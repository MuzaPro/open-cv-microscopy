# Development Plan 05/03/25



## Objectives:

- Make the script work with the new microscope's images
  
  - mark the borders of the EB
  
  - recognize irregular shapes
  
  - analyze fluorescent channels

- Restore the graph generation functionality

- Implement a user-friendly GUI

- Generate clear documentation

- Fix the 'summary image' so it actully shows a grid







# Understanding Why New Microscope Images Aren't Working

We should focus on understanding why the new microscope images aren't working with a script that successfully processes irregular shapes from your old microscope. Rather than simply adjusting parameters, we need to identify the fundamental differences between the image types.

## Diagnostic Strategy

Here's a strategy to diagnose and address the specific issues with your new microscope images:

### 1. Visual Comparison Analysis

First, let's add code to visualize and compare key characteristics between old and new microscope images:

```python
def analyze_image_characteristics(old_image_path, new_image_path):
    """Compare characteristics between old and new microscope images"""
    old_img = cv2.imread(old_image_path, cv2.IMREAD_GRAYSCALE)
    new_img = cv2.imread(new_image_path, cv2.IMREAD_GRAYSCALE)

    # Resize to same dimensions for comparison if needed
    if old_img.shape != new_img.shape:
        new_img = cv2.resize(new_img, (old_img.shape[1], old_img.shape[0]))

    # Create comparison visualizations
    comparison = np.hstack((old_img, new_img))

    # Histograms
    old_hist = cv2.calcHist([old_img], [0], None, [256], [0, 256])
    new_hist = cv2.calcHist([new_img], [0], None, [256], [0, 256])

    # Edge detection comparison
    old_edges = cv2.Sobel(old_img, cv2.CV_64F, 1, 0, ksize=3)
    new_edges = cv2.Sobel(new_img, cv2.CV_64F, 1, 0, ksize=3)
    edge_comparison = np.hstack((old_edges, new_edges))

    # Display results
    plt.figure(figsize=(15, 10))

    plt.subplot(3, 1, 1)
    plt.title("Original Images (Old vs New)")
    plt.imshow(comparison, cmap='gray')

    plt.subplot(3, 1, 2)
    plt.title("Histograms")
    plt.plot(old_hist, color='b', label='Old Microscope')
    plt.plot(new_hist, color='r', label='New Microscope')
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.title("Edge Detection")
    plt.imshow(edge_comparison, cmap='gray')

    plt.tight_layout()
    plt.savefig("microscope_comparison.png")
    plt.show()

    # Print statistical differences
    print(f"Old image - Min: {np.min(old_img)}, Max: {np.max(old_img)}, Mean: {np.mean(old_img):.2f}, StdDev: {np.std(old_img):.2f}")
    print(f"New image - Min: {np.min(new_img)}, Max: {np.max(new_img)}, Mean: {np.mean(new_img):.2f}, StdDev: {np.std(new_img):.2f}")
```

### 2. Pipeline Stage Analysis

Add instrumentation to see where exactly the new images are failing:

```python
def debug_pipeline_stages(image_path, pixel_to_micron):
    """Analyze each stage of the pipeline to identify where failures occur"""
    print(f"\n=== PIPELINE ANALYSIS FOR {os.path.basename(image_path)} ===")
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Track success at each stage
    stage_results = {}

    # Stage 1: Initial processing
    stage_results["original"] = {"image": image.copy(), "success": True}

    # Stage 2: Sobel filter
    sobel_image = process_image_sobel(image_path, os.path.dirname(image_path))
    stage_results["sobel"] = {"image": sobel_image, "success": sobel_image is not None}

    # Stage 3: Gaussian blur
    blur = cv2.GaussianBlur(sobel_image, (7, 7), 3)
    stage_results["blur"] = {"image": blur.copy(), "success": True}

    # Stage 4: Adaptive thresholding
    mask = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, 21, 5)
    stage_results["threshold"] = {"image": mask.copy(), "success": True}

    # Stage 5: Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)
    stage_results["morphology"] = {"image": mask.copy(), "success": True}

    # Stage 6: Contour finding
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
    stage_results["contours"] = {
        "image": contour_img,
        "success": len(contours) > 0,
        "data": f"Found {len(contours)} contours"
    }

    # Stage 7: Size filtering
    suitable_contours = [cnt for cnt in contours if
                        min_size < cv2.contourArea(cv2.convexHull(cnt)) * (pixel_to_micron ** 2) < max_size]
    size_filtered_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(size_filtered_img, suitable_contours, -1, (0, 255, 0), 2)
    stage_results["size_filter"] = {
        "image": size_filtered_img,
        "success": len(suitable_contours) > 0,
        "data": f"Kept {len(suitable_contours)} contours after size filtering"
    }

    # Stage 8: Shape filtering (circularity, solidity, aspect ratio)
    valid_contours = []
    for contour in suitable_contours:
        # Apply contour approximation
        epsilon = 0.005 * cv2.arcLength(contour, True)
        approx_contour = cv2.approxPolyDP(contour, epsilon, True)

        # Calculate circularity
        perimeter = cv2.arcLength(approx_contour, True)
        area = cv2.contourArea(approx_contour)
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0

        # Calculate hull area and solidity
        hull = cv2.convexHull(approx_contour)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area if hull_area > 0 else 0

        # Calculate aspect ratio
        x, y, w, h = cv2.boundingRect(approx_contour)
        aspect_ratio = float(w) / h if h > 0 else 0

        # Check all criteria
        if (circularity >= 0.5 and 0.4 <= solidity <= 1.8 and 
            0.25 <= aspect_ratio <= 4):
            valid_contours.append(approx_contour)

    shape_filtered_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(shape_filtered_img, valid_contours, -1, (0, 255, 0), 2)
    stage_results["shape_filter"] = {
        "image": shape_filtered_img,
        "success": len(valid_contours) > 0,
        "data": f"Kept {len(valid_contours)} contours after shape filtering"
    }

    # Create visualization of all stages
    plt.figure(figsize=(15, 15))
    stages = list(stage_results.keys())
    cols = 2
    rows = (len(stages) + cols - 1) // cols

    for i, stage in enumerate(stages):
        plt.subplot(rows, cols, i + 1)
        stage_data = stage_results[stage]

        if len(stage_data["image"].shape) == 3:
            plt.imshow(cv2.cvtColor(stage_data["image"], cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(stage_data["image"], cmap='gray')

        status = "✓" if stage_data["success"] else "✗"
        title = f"{stage} {status}"
        if "data" in stage_data:
            title += f"\n{stage_data['data']}"
        plt.title(title)
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(f"pipeline_debug_{os.path.basename(image_path)}.png")

    if show_images:
        plt.show()

    # Print detailed filtering info
    if len(suitable_contours) > 0:
        print("\nDetailed contour filtering results:")
        for i, cnt in enumerate(suitable_contours[:5]):  # First 5 contours
            epsilon = 0.005 * cv2.arcLength(cnt, True)
            approx_contour = cv2.approxPolyDP(cnt, epsilon, True)

            perimeter = cv2.arcLength(approx_contour, True)
            area = cv2.contourArea(approx_contour)
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0

            hull = cv2.convexHull(approx_contour)
            hull_area = cv2.contourArea(hull)
            solidity = float(area) / hull_area if hull_area > 0 else 0

            x, y, w, h = cv2.boundingRect(approx_contour)
            aspect_ratio = float(w) / h if h > 0 else 0

            print(f"Contour {i}:")
            print(f"  Area: {area * (pixel_to_micron ** 2):.2f} µm²")
            print(f"  Circularity: {circularity:.2f} {'PASS' if circularity >= 0.5 else 'FAIL'}")
            print(f"  Solidity: {solidity:.2f} {'PASS' if 0.4 <= solidity <= 1.8 else 'FAIL'}")
            print(f"  Aspect ratio: {aspect_ratio:.2f} {'PASS' if 0.25 <= aspect_ratio <= 4 else 'FAIL'}")

    return stage_results
```

### 3. Microscope-Specific Preprocessing Options

Create options tailored to the specific characteristics of each microscope:

```python
def apply_microscope_specific_preprocessing(image, microscope_type="old"):
    """Apply preprocessing specific to microscope type"""
    if microscope_type == "old":
        # Original preprocessing - works well with old microscope
        blur = cv2.GaussianBlur(image, (7, 7), 3)
        mask = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 21, 5)
    else:  # new microscope
        # Add options specifically for new microscope characteristics
        # Option 1: Enhance contrast first
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(image)
        blur = cv2.GaussianBlur(enhanced, (5, 5), 2)  # Less aggressive blur

        # Option 2: Different thresholding strategy
        # Try Otsu's method instead of adaptive thresholding
        _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Option 3: Edge-based approach
        # Use Canny edge detection followed by dilation to close gaps
        # edges = cv2.Canny(blur, 30, 100)
        # kernel = np.ones((5,5), np.uint8)
        # mask = cv2.dilate(edges, kernel, iterations=2)

    return mask
```

### 4. Adaptive Processing Strategy

Implement a system that can automatically detect and handle both microscope types:

```python
def detect_microscope_type(image):
    """Try to automatically detect if image is from old or new microscope"""
    # Calculate image statistics
    mean_val = np.mean(image)
    std_val = np.std(image)

    # Calculate histogram features
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    hist_norm = hist / np.sum(hist)

    # Calculate entropy as a measure of texture
    entropy = -np.sum(hist_norm * np.log2(hist_norm + 1e-7))

    # Check edge characteristics
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    edge_mean = np.mean(edge_magnitude)

    # Make decision based on calculated features
    # These thresholds would need to be calibrated with your actual images
    if edge_mean > 20 and entropy > 5:
        return "old"
    else:
        return "new"
```

## Implementation Plan

1. **Start with diagnostic code** to understand differences:
   
   - Run `analyze_image_characteristics()` on pairs of old/new images that contain similar specimens
   - Use `debug_pipeline_stages()` on both successful and failing images

2. **Common differences to look for**:
   
   - Are the new microscope images higher contrast or lower contrast?
   - Do they have more background noise or less?
   - Are the edges of embryos sharper or more diffuse?
   - Are there illumination inconsistencies?

3. **Once differences are identified**, modify the processing pipeline:
   
   - Either detect microscope type automatically and branch processing
   - Or create a unified pipeline that handles both types with adjustments

4. **Test the modified approach** with both old and new images to ensure backward compatibility

This diagnostic-first approach will help you understand exactly why the new microscope images are failing, rather than just making blind parameter adjustments. Once you understand the fundamental differences, you can implement a targeted solution that works for both image types.
