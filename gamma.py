import numpy as np #for computing numerical operations 
import cv2 #openCV ,it is a computer vision library
from sklearn.cluster import KMeans #for computing k means algorithm 
import os
from scipy import sparse
import time
import os 



def sky_region_segmentation(image, k=2):
    
    image_float32 = image.astype(np.float32) / 255.0 #converting the image into float32 for compatability and normalize the pixel value between [0,1]

    # Flatten the image to perform KMeans clustering 2D 
    pixels = image_float32.reshape((-1, 3))

    # Use KMeans to segment the image
    kmeans = KMeans(n_clusters=k, random_state=0)
    labels = kmeans.fit_predict(pixels)
    #reshape into original image
    segmented_result = labels.reshape(image.shape[:2])

    # Assume the largest cluster corresponds to the sky region
    sky_label = np.argmax(np.bincount(segmented_result.flatten()))
    sky_region_mask = (segmented_result == sky_label)


    return sky_region_mask #return binary mask

def refine_sky_region(image, sky_region_mask):
    #This function refines the sky region mask using OpenCV's guided filter

    # Convert image to float32
    image_float32 = image.astype(np.float32) / 255.0

    # Convert mask to float32
    mask_float32 = sky_region_mask.astype(np.float32)

   #If the image or mask has a single channel, it is converted to a three-channel image.

    if image_float32.shape[-1] == 1:
        image_float32 = cv2.cvtColor(image_float32, cv2.COLOR_GRAY2BGR)
    if mask_float32.shape[-1] == 1:
        mask_float32 = cv2.cvtColor(mask_float32, cv2.COLOR_GRAY2BGR)

    # Use OpenCV's guided filter for refinement
    refined_mask = cv2.ximgproc.guidedFilter(image_float32, mask_float32, radius=5, eps=1e-3)
    
    return refined_mask

def gamma_correction(image, gamma=0.7):
    # Apply gamma correction to the dehazed image

    #Gamma correction is a non-linear adjustment to modify the image intensity.

    corrected_image = np.power(image, gamma)
    return corrected_image

def dehaze_image(image_path):
    # Load of image
    image = cv2.imread(image_path)

    #Loads the input image using OpenCV.
    #Calls the sky_region_segmentation function to obtain a binary mask for the sky region.
    #Calls the refine_sky_region function to refine the sky region using guided filtering.
    #Applies gamma correction to the refined mask using the gamma_correction function.
    #Displays the original and dehazed images using OpenCV.
    
    if image is not None:
        # Step 1: Sky Region Segmentation
        sky_region_mask = sky_region_segmentation(image)

        # Step 2: Guided Filter for the Refinement of Sky Region
        refined_mask = refine_sky_region(image, sky_region_mask)

        # Step 3: Gamma Correction Approach
        corrected_image = gamma_correction(refined_mask, gamma=0.7)  # Adjust gamma value as needed
        
        # Display or save the results as needed
        cv2.imshow("Original Image", image)
        cv2.imshow("Dehazed Image", corrected_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        results_folder = 'results'
        if not os.path.exists(results_folder):
            os.makedirs(results_folder)
        # Construct the output file path
        output_file_path = os.path.join(results_folder, os.path.basename(image_path))
        
        # Save the dehazed image
        cv2.imwrite(output_file_path, corrected_image * 255)  # Make sure to multiply by 255 to convert back to 0-255 range
        print(f"Dehazed image saved at: {output_file_path}")
    else:
        print("Failed to load the image.")
# Example usage
script_path = os.path.abspath(__file__)
data_folder = os.path.join(os.path.dirname(script_path), '_data')
src_path = os.path.join(data_folder, 'content2.png')

# Load the image using cv2.imread()
image = cv2.imread(src_path)

# Call dehaze_image() function with the image file path
dehaze_image(src_path)
