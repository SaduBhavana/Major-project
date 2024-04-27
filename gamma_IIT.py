from sklearn.neighbors import NearestNeighbors
from scipy import sparse
import time
import os

import numpy as np #for computing numerical operations 
import cv2 #openCV ,it is a computer vision library
from sklearn.cluster import KMeans #for computing k means algorithm 

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
    else:
        print("Failed to load the image.")


def image_to_vector(I, param):
    color_transfer = param['color_transfer']
    color_space = param['color_space']
    logarithm = param['logarithm']
    scale = param['scale']
    bias = param['bias']

    rows, cols, layers = I.shape #dimensions of the input 
    Px = np.tile(np.arange(1, rows + 1), (1, cols)) #x coordinates 
    Py = np.tile(np.arange(1, cols + 1), (rows, 1)) # y coordinates 
    P = np.column_stack((Px.flatten(), Py.flatten())) #combine the X and Y coordinates into a single array P.
    # checks if the specified color space is HSV (Hue, Saturation, Value). 
    #If so, it converts the input image I from RGB to HSV color space using cv2.cvtColor().
    # Then it extracts the V (value/brightness) channel. 
    #If color_transfer is True, it extracts only the H and S channels from the HSV image
    # otherwise, it applies logarithmic transformation  to the V channel and flattens it into a 1D array X.
    if color_space == 'hsv':  
        hsv_I = cv2.cvtColor(I, cv2.COLOR_RGB2HSV)
        v = hsv_I[:, :, 2]
        if color_transfer:
            X = hsv_I[:, :, :2].reshape((rows * cols, 2))
        else:
            if logarithm:
                v = -np.log(bias + scale * v)
            X = v.flatten()
    #It converts the input image I from RGB to Lab color space using cv2.cvtColor(). 
    #Then it extracts the L (lightness) channel and scales it to the range [0, 1]
    elif color_space == 'lab':
        Lab_I = cv2.cvtColor(I, cv2.COLOR_RGB2LAB)
        L = Lab_I[:, :, 0] / 100.0
        if color_transfer:
            X = Lab_I[:, :, 1:].reshape((rows * cols, 2))
        else:
            if logarithm:
                L = -np.log(bias + scale * L)
            X = L.flatten()
    #Then it extracts the Y (luminance) channel.
    #Depending on the value of color_transfer, it either extracts the Cb and Cr channels or applies a logarithmic transformation.
    elif color_space == 'Ycbcr':
        Ycbcr_I = cv2.cvtColor(I, cv2.COLOR_RGB2YCrCb)
        L = Ycbcr_I[:, :, 0]
        if color_transfer:
            X = Ycbcr_I[:, :, 1:].reshape((rows * cols, 2))
        else:
            if logarithm:
                L = -np.log(bias + scale * L)
            X = L.flatten()
    else:
        if logarithm:
            I = -np.log(bias + scale * I)
        X = I.reshape((rows * cols, layers))

    return X, P #returns flattened pixel values X and the pixel coordinates P.



def solve_gaussian_weights(X, P, K, delta_s, delta_r, mode, scale, bias):
    N, _ = X.shape # Extracts the number of samples N from the first dimension of the feature vectors X.
    nn = NearestNeighbors(n_neighbors=K)
    nn.fit(P) # Fits the nearest neighbors model with the spatial positions P
    distances, indices = nn.kneighbors(P, return_distance=True)

    X = (np.exp(-X) - bias) / scale
    p = P[indices[:, 1:], :] - P[indices[:, :1], :] #Computes the spatial differences between the neighbors for each sample.
    x = X[indices[:, 1:], :] - X[indices[:, :1], :] #. It subtracts the feature vector of the first neighbor from the feature vectors of the other neighbors.

    if mode == 'gf':
        delta_s2 = 2 * delta_s**2 # Computes the squared spatial bandwidth.
        w = np.exp(-(np.sum(p**2, axis=2) / delta_s2)) #Computes the Gaussian weights based on the spatial differences.
    elif mode == 'bf':
        delta_s2 = 2 * delta_s**2 #computes the squared spatial bandwidth.
        delta_r2 = 2 * delta_r**2
        w = np.exp(-(np.sum(p**2, axis=2) / delta_s2) - (np.sum(x**2, axis=2) / delta_r2)) #Computes the Gaussian weights based on both spatial and feature differences.

    else:
        delta_s2 = 2 * delta_s**2
        delta_r2 = 2 * delta_r**2
        w = np.exp(-(np.sum(p**2, axis=2) / delta_s2) - (np.sum(x**2, axis=2) / delta_r2))

    w /= np.sum(w, axis=1)[:, np.newaxis] #Normalizes the weights along each row. This ensures that the weights for each sample sum to 1.

    # Create COO sparse matrix
    row_idx = np.repeat(np.arange(N), K-1)
    col_idx = indices[:, 1:].flatten()
    data = w.flatten()

    if len(row_idx) == len(col_idx) == len(data) == N * (K - 1):
        W_coo = sparse.coo_matrix((data, (row_idx, col_idx)), shape=(N, N), dtype=np.float64)
        
        # Convert to CSR sparse matrix
        W = W_coo.tocsr() #parse matrix representation of the computed weights using COO format. This format stores the (row, column, value)
        M = W.transpose().dot(W)

        return M # affinity matrix M, representing the pairwise similarities between samples based on their feature and spatial similarities.
    else: 
        raise ValueError("Length mismatch in COO matrix creation.")

def solve_lle_embedding(C, S, W, M, alpha, beta, gamma):
    n, _ = M.shape #Extracts the number of rows  from the affinity matrix M.
    m, d = C.shape #Extracts the number of rows and columns from the data matrix C

    A = alpha * W + beta * M + gamma * sparse.eye(m, format='csr') # captures the local structure of the data.
    b = alpha * W.dot(C) + gamma * S #captures the global structure of the data.
    Y = sparse.linalg.spsolve(A, b)

    return Y #the low-dimensional representation of the data 

def solve_lle_weights(X, P, K, tol):

    N, D = P.shape

    nn = NearestNeighbors(n_neighbors=K + 1)
    nn.fit(P)
    distances, indices = nn.kneighbors(P, return_distance=True)
    neighborhood = indices[:, 1:]
    currentindex = indices[:, :1]

    w0 = np.zeros((N, K))
    for ii in range(N):
        z = X[currentindex[ii, :], :] - X[neighborhood[ii, :], :]
        c = z.dot(z.T)
        c = c + tol * np.eye(K)
        w = np.linalg.solve(c, np.ones(K))
        w0[ii, :] = w / np.sum(w)

    # Ensure all arrays have the same length
    common_length = min(len(w0.ravel()), len(currentindex.flatten()), len(neighborhood.flatten()))
    
   # W_coo = sparse.coo_matrix((w0.ravel()[:common_length], (currentindex.flatten()[:common_length], neighborhood.flatten()[:common_length])), shape=(N, N))
    W_coo = sparse.coo_matrix((w0.ravel()[:common_length], (currentindex.flatten()[:common_length], neighborhood.flatten()[:common_length])), shape=(N, N))

    # Only return the necessary value
    return W_coo

def vector_to_image(Y, S, C, param):
    bias = param['bias'] # Bias used during vector-to-image transformation.
    scale = param['scale'] #Scaling factor used during vector-to-image transformation.
    color_space = param['color_space'] #Specifies the color space of the output image.
    logarithm = param['logarithm'] #xtracts the boolean flag indicating whether to apply logarithmic transformation from the parameter dictionary.
    color_transfer = param['color_transfer'] #perform color transfer from the parameter dictionary.
    color_exemplar = param['color_exemplar'] #Extracts the information about which image to use as an exemplar for color transfer from the parameter dictionary.

    rows, cols, layers = S.shape

    if color_space == 'hsv':
        if color_exemplar == 'original':
            hsv_S = cv2.cvtColor(S, cv2.COLOR_RGB2HSV)
        elif color_exemplar == 'exemplar':
            hsv_S = cv2.cvtColor(C, cv2.COLOR_RGB2HSV)

        if color_transfer:
            hsv_S[:, :, :2] = Y.reshape((rows, cols, 2))
        else:
            v = Y
            if logarithm:
                v = np.minimum(1, np.maximum(0, (np.exp(-v) - bias) / scale))
            hsv_S[:, :, 2] = v.reshape((rows, cols, 1))
        I = cv2.cvtColor(hsv_S, cv2.COLOR_HSV2RGB)
    elif color_space == 'lab':
        if color_exemplar == 'original':
            lab_S = cv2.cvtColor(S, cv2.COLOR_RGB2LAB)
        elif color_exemplar == 'exemplar':
            lab_S = cv2.cvtColor(C, cv2.COLOR_RGB2LAB)
        if color_transfer:
            lab_S[:, :, 1:] = Y.reshape((rows, cols, 2))
        else:
            L = Y
            if logarithm:
                L = np.minimum(1, np.maximum(0, (np.exp(-L) - bias) / scale))
            lab_S[:, :, 0] = L.reshape((rows, cols, 1))
        I = cv2.cvtColor(lab_S, cv2.COLOR_LAB2RGB)
    elif color_space == 'Ycbcr':
        if color_exemplar == 'original':
            Ycbcr_S = cv2.cvtColor(S, cv2.COLOR_RGB2YCrCb)
        elif color_exemplar == 'exemplar':
            Ycbcr_S = cv2.cvtColor(C, cv2.COLOR_RGB2YCrCb)
        if color_transfer:
            Ycbcr_S[:, :, 1:] = Y.reshape((rows, cols, 2))
        else:
            L = Y
            if logarithm:
                L = np.minimum(1, np.maximum(0, (np.exp(-L) - bias) / scale))
            Ycbcr_S[:, :, 0] = L.reshape((rows, cols, 1))
        I = cv2.cvtColor(Ycbcr_S, cv2.COLOR_YCrCb2RGB)
    else:
        if logarithm:
            Y = np.minimum(1, np.maximum(0, (np.exp(-Y) - bias) / scale))
        I = Y.reshape((rows, cols, layers))

    return I

def intrinsic_image_transfer(S, C, param):
    X_s, P = image_to_vector(S, param)
    X_c, _ = image_to_vector(C, param)

    bias = param['bias']
    scale = param['scale']
    
    k1 = param['filter']['k1']
    delta_s = param['filter']['delta_s']
    delta_r = param['filter']['delta_r']
    filter_mode = param['filter']['mode']
    W_coo = solve_gaussian_weights(X_s, P, k1, delta_s, delta_r, filter_mode, scale, bias)

    tol = param['LLE']['tol']
    k2 = param['LLE']['k2']
    
    # Use the coo_matrix directly
    M = W_coo.transpose().dot(W_coo)

    alpha = param['alpha']
    beta = param['beta']
    gamma = param['gamma']
    Y = solve_lle_embedding(X_c, X_s, W_coo, M, alpha, beta, gamma)

    T = vector_to_image(Y, S, S, param)
    T = np.maximum(0, np.minimum(1, T))

    return T, M #ntrinsic image T and the affinity matrix M.


def main():
    start_time = time.time()
    print(start_time)
    # Get the absolute path of the script
    script_path = os.path.abspath(__file__)

    # Assume the 'data' folder is in the same directory as the script
    data_folder = os.path.join(os.path.dirname(script_path), '_data')

    src_path = os.path.join(data_folder, 'content2.png')


    demo_folder=os.path.join(os.path.dirname(script_path),'_data_')

    
    clahe_path = os.path.join(demo_folder, 'exemplar2.png')

    # Check if the files exist
    if not os.path.isfile(src_path) or not os.path.isfile(clahe_path):
        print(f"Error: One or both images do not exist.")
        return

    # Load images
    S = cv2.imread(src_path) / 255.0
    C = cv2.imread(clahe_path) / 255.0
    S = cv2.resize(S, (new_width, new_height))
    C = cv2.resize(C, (new_width, new_height))

    # Check if images were loaded successfully
    if S is None or C is None:
        print("Error: One or both images could not be loaded.")
        return
    # Perform sky region segmentation on the source image
    sky_region_mask = sky_region_segmentation((S * 255).astype(np.uint8))

    # Refine the sky region mask using guided filter
    refined_mask = refine_sky_region(S, sky_region_mask)

    # Apply gamma correction to the refined mask
    corrected_mask = gamma_correction(refined_mask, gamma=0.7)

    param = {
        'logarithm': 1, # 1 suggesting that logarithmic transformation is enabled
        'color_transfer': 0, #meaning color transfer is disabled.
        'bias': 1 / 255,  # for 8-bit image
        'scale': 1.0, #Represents the scaling factor used in image transformation
        'color_space': 'rgb', #Specifies the color space used in image processing
        'color_exemplar': 'original',
        'filter': {
            'k1': 49, #Specifies the number of nearest neighbors used in weight computation.
            'delta_s': 2.0, #Represents the spatial standard deviation in the Gaussian filter
            'delta_r': 0.2, #epresents the range standard deviation in the Gaussian filte
            'mode': 'gf' 
        },
        'LLE': {
            'tol': 1e-5, #Specifies the tolerance value used in LLE computations.
            'k2': 49 #Specifies the number of nearest neighbors used in LLE computations.
        },
        'alpha': 0.9, #Represents the weighting factor for the affinity matrix in LLE computations.
        'beta': 100, #Represents the weighting factor for the similarity matrix in LLE computations.
        'gamma': 0.1 #Represents the regularization parameter in LLE computations.
    }

    T, M = intrinsic_image_transfer(S, C, param)
    # T = np.maximum(0, np.minimum(1, T))
   # corrected_mask_resized = cv2.resize(corrected_mask, (T.shape[1], T.shape[0]))

    # Replicate the single channel of the mask to match the three channels of the image
    #corrected_mask_3ch = np.stack([corrected_mask_resized] * 3, axis=-1)

    # Ensure that both arrays have the same shape before multiplication
    #T = np.maximum(0, np.minimum(1, T * corrected_mask_3ch))
    end_time = time.time()
    print(end_time)
    # Save the result
    result_folder = os.path.join(os.path.dirname(script_path), 'results')
    os.makedirs(result_folder, exist_ok=True)
    # Before saving the result, add debug statements to check the values of T and corrected_mask
    print("T min/max:", np.min(T), np.max(T))
    print("corrected_mask min/max:", np.min(corrected_mask), np.max(corrected_mask))

    cv2.imwrite(os.path.join(result_folder, 'IIT_src.png'), (S * 255).astype(np.uint8))
    cv2.imwrite(os.path.join(result_folder, 'IIT_output.png'), (C * 255).astype(np.uint8))

    print(f"Processing time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    new_width, new_height = 500, 500
    main()