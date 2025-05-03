# step 1: Diagnostic Script

## terminal output
PS C:\Users\mrgav\Documents\GitHub\open-cv-microscopy> & C:/Users/mrgav/AppData/Local/Programs/Python/Python310/python.exe c:/Users/mrgav/Documents/GitHub/open-cv-microscopy/ImageAnalysis/development/microscope_image_comparison.py
Analyzing images:
- Old: ImageAnalysis\NIS\Msgn1_72h\exp15_metabolism_mp1_3d_day3_72hc1xy01.tif
- New: ImageAnalysis\Zeiss\Msgn1_98h\exp15_Metabolism_MP1_3D_Day4_98h_b0s6c2x42275-1388y70216-1040.tif
Successfully loaded both images

=== Basic Image Statistics ===
Old image - Min: 0, Max: 255, Mean: 137.56, StdDev: 44.17
New image - Min: 0, Max: 200, Mean: 80.91, StdDev: 32.93
Saved original image comparison
Saved histogram comparison
Saved edge detection comparison
Saved threshold comparison
Saved adaptive threshold comparison

Analysis complete! Review the generated images in the output folder.
Look for differences in contrast, brightness, edge clarity, and noise levels.


## Human input reviewing the images that were generated:

clear differences in the histogram comparison 