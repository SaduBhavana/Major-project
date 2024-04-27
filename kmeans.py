import numpy as np #for computing numerical operations 
import cv2 #openCV ,it is a computer vision library
from sklearn.cluster import KMeans #for computing k means algorithm 
import os
from scipy import sparse
import time
import os 

def visualize_clustered_image(image, labels):
    # Convert labels to uint8 for visualization
    labels_uint8 = labels.astype(np.uint8)

    # Assign a unique color to each label
    colors = [np.random.randint(0, 256, 3) for _ in range(len(np.unique(labels)))]

    # Create a blank image to store the segmented result
    segmented_image = np.zeros_like(image)

    # Fill the segmented image with colors based on labels
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            segmented_image[i, j] = colors[labels_uint8[i, j]]

    # Display the segmented image
    cv2.imshow("Segmented Image", segmented_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def sky_region_segmentation(image, k=2):
    # Convert image to float32
    image_float32 = image.astype(np.float32) / 255.0

    # Flatten the image to perform KMeans clustering
    pixels = image_float32.reshape((-1, 3))

    # Use KMeans to segment the image
    kmeans = KMeans(n_clusters=k, random_state=0)
    labels = kmeans.fit_predict(pixels)

    # Reshape labels to original image shape
    segmented_result = labels.reshape(image.shape[:2])

    # Visualize the clustered image
    visualize_clustered_image(image, segmented_result)

# Example usage
script_path = os.path.abspath(__file__)
data_folder = os.path.join(os.path.dirname(script_path), '_data')
src_path = os.path.join(data_folder, 'content2.png')
image = cv2.imread(src_path)
sky_region_segmentation(image)
