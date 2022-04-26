import numpy as np

import cv2 as cv

"""
Preprocessing
"""
def automatic_brightness_contrast(bgr_image, clip_hist_percent = 0.01, use_scale_abs = True, return_verbose = False):
    gray_image = cv.cvtColor(bgr_image, cv.COLOR_BGR2GRAY)

    # Grayscale histogram of the image
    hist = cv.calcHist([gray_image], [0], None, [256], [0, 256])
    hist_size = len(hist)

    # Cumulative distribution of the histogram
    acc = []
    acc.append( float(hist[0]) )
    for i in range(1, hist_size):
        acc.append( acc[i - 1] + float(hist[i]) )
    
    # Locate points to clip
    maximum = acc[-1]
    clip_hist = clip_hist_percent * maximum / 2.0

    # Left cut
    minimum_gray = 0
    while acc[minimum_gray] < clip_hist:
        minimum_gray += 1

    # Right cut
    maximum_gray = hist_size - 1
    while acc[maximum_gray] >= (maximum - clip_hist):
        maximum_gray -= 1

    # Calculate alpha and beta values for the scaling
    alpha = 255 / (maximum_gray - minimum_gray)
    beta = - minimum_gray * alpha

    if use_scale_abs:
        processed_image = cv.convertScaleAbs(bgr_image, alpha=alpha, beta=beta)
    else:
        processed_image = bgr_image * alpha + beta
        processed_image[processed_image < 0] = 0
        processed_image[processed_image > 255] = 255

    if return_verbose:
        processed_hist = cv.calcHist([gray_image], [0], None, [256], [minimum_gray, maximum_gray])

        return processed_image, alpha, beta, hist, processed_hist
    
    return processed_image

def hsv_clahe_equalization(bgr_image, clipLimit = 2.0, tileGridSize = (8, 8)):
    hsv_image = cv.cvtColor(bgr_image, cv.COLOR_BGR2HSV)

    h, s, v = cv.split(hsv_image)

    clahe = cv.createCLAHE(
        clipLimit=clipLimit,
        tileGridSize=tileGridSize
    )

    s_equalized = clahe.apply(s)
    v_equalized = clahe.apply(v)

    equalized_image = cv.merge([h, s_equalized, v_equalized])
    return cv.cvtColor(equalized_image, cv.COLOR_HSV2BGR)

def LaplacianOfGaussian(bgr_image):
    log_image = cv.GaussianBlur(bgr_image, (3, 3), 0)
    gray_image = cv.cvtColor(log_image, cv.COLOR_BGR2GRAY)
    log_image = cv.Laplacian(gray_image, cv.CV_8U, ksize=3, scale=1, delta=0)
    return cv.convertScaleAbs(log_image)

"""
Segmentation
"""
def meanShiftFiltering(bgr_image):
    return cv.pyrMeanShiftFiltering(bgr_image, 10, 15, 100)

def binarize(bgr_image, threshold_percent = 0.75, return_thresh = False):
    gray_image = cv.cvtColor(bgr_image, cv.COLOR_BGR2GRAY)

    threshold = int(threshold_percent * 255)

    thresh, thresh_image = cv.threshold(gray_image, threshold, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    if return_thresh:
        return thresh_image, thresh
    return thresh_image

def removeSmallComponents(binary_image, size_threshold = 300):
    n_components, output, stats, _ = cv.connectedComponentsWithStats(binary_image, connectivity=8)
    sizes = stats[1:, -1]
    out_image = np.zeros(output.shape, dtype=np.uint8)
    for i in range(n_components - 1):
        if sizes[i] >= size_threshold:
            out_image[output == i + 1] = 255
    
    return out_image

def extract_red_hsv(hsv_image, return_split = False):
    # First zone of reds
    lowerbound_1 = np.array([0, 40, 40])
    upperbound_1 = np.array([10, 255, 255])

    # Second zone of reds
    lowerbound_2 = np.array([135, 40, 40])
    upperbound_2 = np.array([179, 255, 255])

    red_1 = cv.inRange(hsv_image, lowerbound_1, upperbound_1)
    red_2 = cv.inRange(hsv_image, lowerbound_2, upperbound_2)

    if return_split:
        return red_1, red_2
    
    return cv.bitwise_or(red_1, red_2)

def extract_blue_hsv(hsv_image):
    lowerbound = np.array([100, 128, 40])
    upperbound = np.array([120, 255, 255])

    return cv.inRange(hsv_image, lowerbound, upperbound)

def extract_white_hsv(hsv_image):
    lowerbound = np.array([0, 0, 127])
    upperbound = np.array([0, 40, 255])

    return cv.inRange(hsv_image, lowerbound, upperbound)

"""
Shape detection
"""
def getCountours(image, return_hierarchy=False):
    contours, hierarchy = cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    
    if return_hierarchy:
        return contours, hierarchy
    return contours

def extractObjects(bgr_image, contours):
    objects = []

    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)
        objects.append(bgr_image[y:y+h, x:x+w])
    
    return objects