# import the cv2 as well as numpy library
import cv2
import numpy as np
import os
from skimage.metrics import structural_similarity as ssim
# Create a function that takes two images’ paths as a parameter
def calculate_psnr(firstImage, secondImage):
   # Compute the difference between corresponding pixels
   diff = np.subtract(firstImage, secondImage)
   # Get the square of the difference
   squared_diff = np.square(diff)

   # Compute the mean squared error
   mse = np.mean(squared_diff)

   # Compute the PSNR
   max_pixel = 255
   psnr = 20 * np.log10(max_pixel) - 10 * np.log10(mse)
    
   return psnr

# Resize images to a common size
rHeight = 256
rWidth = 256

# Read the original and distorted images
script_path = os.path.abspath(__file__)
data_folder = os.path.join(os.path.dirname(script_path), 'results')

 # Use absolute paths for the images
src_path = os.path.join(data_folder, 'IIT_output.png')
des_path=os.path.join(data_folder,'IIT_src.png')

firstI = cv2.imread(des_path)
secondI = cv2.imread(src_path)

# Check if images are loaded successfully
if firstI is None or secondI is None:
   print("Failed to load one or both images.")
else:
   # Resize images for first image
   firstI = cv2.resize(firstI, (rWidth, rHeight))
   # Resize the details for second image
   secondI = cv2.resize(secondI, (rWidth, rHeight))
    
   # Call the above function and perform the calculation
   psnr_score = calculate_psnr(firstI, secondI)
   #ssim_index, _ = ssim(firstI, secondI, full=True)
   win_size = min(firstI.shape[0], firstI.shape[1]) // 100  # Example: Using 1/100th of the smaller dimension
   win_size = win_size + 1 if win_size % 2 == 0 else win_size  # Ensure win_size is odd
   ssim_index, _ = ssim(firstI, secondI, full=True, win_size=win_size)
   print("SSIM:", ssim_index)
   # Display the result
   print("PSNR:", psnr_score)