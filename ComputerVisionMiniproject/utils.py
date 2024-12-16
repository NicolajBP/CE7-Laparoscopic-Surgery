from skimage.feature import hog as HOG
import numpy as np
import cv2 as cv
from skimage.exposure import rescale_intensity
from matplotlib import pyplot as plt


# extract features from patch of image
def extract_features(patch):
    # compute HoG descriptors
    winSize = (128, 64) if patch.shape[1] > patch.shape[0] else (64, 128) # (w, h)
    cellSize = (6, 6) # (w, h)
    cellsBlockSize = (2, 2) # (w, h)
    blockSize = (16, 16) # (w, h)
    blockStride = (8, 8) # (w, h)
    numBins = 9
    
    hog_features_desc, hog_features = HOG(patch, orientations=numBins, pixels_per_cell=cellSize,
                        cells_per_block=cellsBlockSize, visualize=True, feature_vector=False, channel_axis=2)

    return hog_features_desc, hog_features

# initialize filter
def init_filter(features):
    # define the features as filter in FFT domain
    filter = features
    return filter

# apply filter on features image
def apply_filter(filter, features):
    # apply correlation filter
    h, w = filter.shape[:2]
    h_f, w_f = features.shape[:2]
    
    # res = cv.matchTemplate(np.mean(features, axis=(2,3,4)).astype('float32'), 
    #                        np.mean(filter, axis=(2,3,4)).astype('float32'),
    #                        cv.TM_CCOEFF_NORMED)

    
    # appy in FFT domain (multiply filter by conjugate of features in FFT is same as spatial correlation)
    filter = np.reshape(filter, (h,w,-1))
    K = filter.shape[-1]
    features = np.reshape(features, (h_f,w_f,-1))
    filter_fft = np.zeros_like(features)
    features_fft = np.zeros_like(features)
    # res = np.zeros((abs(h_f-h)+1, abs(w_f-w)+1, K))
    res = np.zeros((h, w, K))
    for k in range(K):
        filter_fft[:,:,k] = cv.resize(cv.dft(filter[:,:,k], cv.DFT_COMPLEX_OUTPUT), features.shape[1::-1])
        features_fft[:,:,k] = cv.dft(features[:,:,k], cv.DFT_COMPLEX_OUTPUT)
        
        res_fft = cv.mulSpectrums(filter_fft[:,:,k], features_fft[:,:,k], 0, conjB=True)
        res[:,:,k] = cv.resize(cv.idft(res_fft, cv.DFT_SCALE | cv.DFT_REAL_OUTPUT), (w, h))
        
    return np.mean(np.abs(res), axis=(2))

# update the filter using learning
def update_filter(filter, new_features, learning_rate=0.01):
    # update filter with the fft of new features (as it's the definition of the filter) by exponential update
    filter = new_features * learning_rate + filter * (1 - learning_rate)
    return filter

# visualize the HOG descriptors image
def visualize_HOG(image, hog_image, cv_flag=False):
    # Rescale the hog_image for better contrast
    hog_image_rescaled = rescale_intensity(hog_image, in_range=(0, 10))

    if cv_flag:
        # show in opencv imshow
        cv.imshow("HoG Visualization", np.hstack((cv.cvtColor(image, cv.COLOR_BGR2GRAY), 255*hog_image_rescaled.astype('uint8'))))
    else:
        # Plot original image and HoG visualization
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.title('Original Image')
        plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))

        plt.subplot(1, 2, 2)
        plt.title('HoG Visualization')
        plt.imshow(hog_image_rescaled, cmap='gray')

        plt.show()

# rescale patches using pre-defined scales
def get_rescaled_patches(frame, bbox, scales):
    rescaled_patches = []
    x, y, w, h = bbox
    cx = x + w//2
    cy = y + h//2 # centers of bbox
    for scale in scales:
        # Adjust the bounding box size based on the scale
        scaled_w = int(w * scale)
        scaled_h = int(h * scale)
        
        # Extract the patch from the frame using the rescaled bounding box
        patch = frame[(cy-scaled_h//2):(cy+scaled_h//2), (cx-scaled_w//2):(cx+scaled_w//2)]
        rescaled_patches.append(patch)
    return rescaled_patches

# find the best correlated scale
def find_best_scale(filter, rescaled_patches):
    best_response = None
    best_scale_idx = 0
    for idx, patch in enumerate(rescaled_patches):
        features_hog_desc, features_hog = extract_features(patch)  # Extract features from each rescaled patch
        response = apply_filter(filter, features_hog_desc)  # Apply the correlation filter

        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(response)
        if best_response is None or max_val > np.max(best_response):
            best_response = response
            best_scale_idx = idx
    
    return best_scale_idx

# update boundin box scale
def update_bbox_scale(bbox, best_scale_idx, scales):
    x, y, w, h = bbox
    best_scale = scales[best_scale_idx]
    # Update width and height based on the selected scale
    new_w = int(w * best_scale)
    new_h = int(h * best_scale)
    new_x = x + w//2 - new_w//2
    new_y = y + h//2 - new_h//2
    return (new_x, new_y, new_w, new_h)


# create epanechnikov kernel for the foreground estmation
def create_epanechnikov_kernel(size):
    h, w = size
    center_y, center_x = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    X, Y = np.meshgrid(x, y)
    
    # Distance from center
    dist = (X - center_x) ** 2 / ((w/2)**2)/2 + \
        (Y - center_y) ** 2 / ((h/2)**2)/2
    kernel = 0.75 * (1 - dist)  # Parabolic shape
    kernel[dist > 1] = 0  # Mask out distances greater than 1 (outside the ellipse)
    
    return kernel

# get histogram of image
def compute_histogram(image, mask=None, bins=(16, 16, 16)):
    # Convert image to HSV color space for histogram calculation
    hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    
    # Compute color histogram
    hist = cv.calcHist([hsv_image], [0, 1, 2], mask, bins, [0, 180, 0, 256, 0, 256])
    cv.normalize(hist, hist, 0, 255, cv.NORM_MINMAX)
    
    return hist

# extract foreground background histograms from bbox
def extract_fg_bg_histograms(image, bbox):
    x, y, w, h = bbox
    
    # Extract the foreground patch within the bounding box
    foreground_patch = image[y:y+h, x:x+w]
    
    # Create the Epanechnikov kernel for the foreground
    kernel = create_epanechnikov_kernel((h, w))
    
    # Create the foreground mask using the kernel
    foreground_mask = (kernel * 255).astype(np.uint8)
    
    # Define the background region as twice the bounding box size
    bg_x1 = max(0, x - w // 2)
    bg_y1 = max(0, y - h // 2)
    bg_x2 = min(image.shape[1], x + w + w // 2)
    bg_y2 = min(image.shape[0], y + h + h // 2)
    
    background_patch = image[bg_y1:bg_y2, bg_x1:bg_x2]
    
    # Compute the foreground and background histograms
    fg_hist = compute_histogram(foreground_patch, foreground_mask)
    bg_hist = compute_histogram(background_patch)
    
    return fg_hist, bg_hist

# calculate colors probability for foreground background
def calculate_color_probability(image, hist, bins):
    # Convert the image to HSV for comparison
    hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    
    # Flatten the image for easy histogram lookup
    pixel_values = hsv_image.reshape((-1, 3))
    
    # Get the bin indexes for each pixel in HSV space
    bin_indices = np.floor(pixel_values / np.array([180 / bins[0], 256 / bins[1], 256 / bins[2]])).astype(int)
    
    # Clip bin indices to avoid out-of-range errors
    bin_indices = np.clip(bin_indices, 0, np.array(bins) - 1)
    
    # Look up probabilities in the histogram
    probs = hist[bin_indices[:, 0], bin_indices[:, 1], bin_indices[:, 2]]
    
    # Reshape the probabilities to match the original image
    return probs.reshape(hsv_image.shape[:2])

# compute the spatial reliability for foreground relative to background
def compute_spatial_reliability(fg_prob_map, bg_prob_map):
    # Add a small constant to avoid division by zero
    spatial_reliability = fg_prob_map / (fg_prob_map + bg_prob_map + 1e-6)
    
    return spatial_reliability

# apply spatial map to features
def apply_spatial_map_to_features(features, spatial_map):
    # Resize the spatial reliability map to match the feature map size (if needed)
    if features.shape[:2] != spatial_map.shape:
        spatial_map = cv.resize(spatial_map, (features.shape[1], features.shape[0]))
    
    # Apply the spatial reliability map by element-wise multiplication
    return features * spatial_map