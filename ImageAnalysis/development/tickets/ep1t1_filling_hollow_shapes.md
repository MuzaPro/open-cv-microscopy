# Development Ticket: Enhance Embryo Detection for New Microscope Images

**Ticket ID:** EMBD-101  
**Priority:** High  
**Assigned to:** Junior Developer  
**Estimated time:** 4-6 hours

## Problem Description

Our current image processing script (`spheroid_analyzer.py`) works well with old microscope images but fails to properly detect embryo bodies from the new Zeiss microscope. Analysis shows this is because:

1. The new microscope produces images with lower contrast and different intensity distribution
2. Thresholding creates "hollow ring" shapes instead of solid shapes for embryos
3. The script is detecting small fragments of the boundary rather than complete embryos

## Objective

Modify a copy of the script to implement a specific "fill hollow shapes" preprocessing step for the new microscope images.

## Implementation Steps

### 1. Create a Modified Version of the Script

We've already created a copy of the script. Rename it to `image_analysis_revized.py`.

### 2. Add Configuration Parameter

At the top of the file in the configuration section, add this parameter:
```python
# Add this below the other configuration parameters
microscope_type = "new"  # Options: "old" or "new"
```

### 3. Modify the `process_contours` Function

Replace the existing `process_contours` function with this modified version that includes the hollow shape filling functionality:

```python
def process_contours(image, image_path, results_df, min_area_um2, max_area_um2, pixel_to_micron=pixel_to_micron):
    print(f"Processing contours in: {image_path}")
    image_name = os.path.basename(image_path)
    added_valid_contour = False  # Track if we added a valid contour
    
    # Apply Gaussian blur
    blur = cv2.GaussianBlur(image, (7, 7), 3)
    if show_images:
        image_collector.add(blur, f'Gaussian Blur: {image_name}')
    
    # Apply microscope-specific preprocessing
    if microscope_type == "new":
        # For new microscope: Use Otsu's thresholding which works better with bimodal histograms
        _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Create a copy for visualization before filling
        pre_fill_mask = mask.copy()
        if show_images:
            image_collector.add(pre_fill_mask, f'Pre-Fill Mask: {image_name}')
        
        # --- Step 1: Fill holes in the mask using contour filling ---
        fill_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw filled contours on a new mask
        filled_mask_1 = np.zeros_like(mask)
        for contour in fill_contours:
            # Only fill contours that are likely to be embryos (have reasonable size)
            contour_area = cv2.contourArea(contour)
            if contour_area > (min_area_um2 / (pixel_to_micron ** 2) / 10):
                cv2.drawContours(filled_mask_1, [contour], 0, 255, -1)  # -1 means fill
        
        # --- Step 2: Use floodfill as an alternative fill method ---
        h, w = mask.shape
        fill_mask = np.zeros((h+2, w+2), np.uint8)  # +2 for border padding required by floodFill
        mask_copy = mask.copy()
        cv2.floodFill(mask_copy, fill_mask, (0, 0), 0)  # Fill from corner (background)
        filled_mask_2 = mask | cv2.bitwise_not(mask_copy)
        
        # --- Step 3: Combine both filling approaches ---
        mask = filled_mask_1 | filled_mask_2
        
        if show_images:
            image_collector.add(filled_mask_1, f'Contour Fill: {image_name}')
            image_collector.add(filled_mask_2, f'FloodFill: {image_name}')
            image_collector.add(mask, f'Combined Fill: {image_name}')
        
    else:
        # Original processing for old microscope
        mask = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 21, 5)
    
    if show_images:
        image_collector.add(mask, f'Final Mask: {image_name}')
    
    # Continue with the original morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
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
    
    # Continue with the existing contour detection and processing
    # (The rest of the function remains unchanged)
    
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
        # The rest of your existing contour processing code remains unchanged...
        
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
    
    # Continue with the rest of your existing function...
    
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
        # The original multi-contour processing code remains unchanged...
        
    # If we're looking for one contour but didn't add any valid ones
    if one_contour and not added_valid_contour:
        print(f"No valid contours processed in {image_path}. Adding empty line.")
        empty_line = pd.DataFrame([{col: 0 for col in results_df.columns}], index=[0])
        results_df = pd.concat([results_df, empty_line], ignore_index=True)
    
    return results_df
```

### 4. Testing Instructions

1. Ensure you have both old and new microscope test images available
2. Edit the script's configuration section to set the correct paths:
   ```python
   main_folder_path = r'ImageAnalysis\development\investigation_output\filling_output\Zeiss_test.xlsx'  # Point to new microscope test images
   output_file_path = r'ImageAnalysis\Zeiss\Msgn1_98h'
   
   # Make sure these are set appropriately for testing
   show_images = True
   one_contour = True
   ```

3. Run the script on a single directory with new microscope images
4. Check the processed_images folder that will be created to verify that:
   - The "Pre-Fill Mask" shows hollow rings
   - The "Combined Fill" shows solid filled shapes
   - The "Contours" image should show green outlines around complete embryos

5. If the filling appears to work but contours are still not being detected properly, verify:
   - The filled shapes are large enough (check against min_size)
   - The circularity, solidity, and aspect ratio requirements are being met

### 5. Expected Results

1. The script should now detect complete embryo bodies in new microscope images
2. The generated Excel file should contain valid measurements
3. The visualizations should show green contours around complete embryos

### 6. Additional Notes

- Make sure you test with `show_images = True` to verify each step of the processing
- If the fill approach doesn't work well, we may need to try alternate solutions
- Do not modify any parameters like circularity or solidity thresholds yet - focus only on implementing the fill method

## Deliverables

1. Modified script (`image_analysis_revized.py`)
2. Brief report on test results
3. Sample output images showing successful detection (if any)

Let me know if you have any questions about the implementation.