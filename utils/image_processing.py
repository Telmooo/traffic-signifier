import numpy as np

import cv2 as cv

"""
Utility
"""
def clamp(value, minimum, maximum):
    return min(maximum, max(minimum, value))

def weighted_gray_transform(bgr_image, weights=[0.114, 0.587, 0.299]):
    m = np.array(weights).reshape((1,3))
    return cv.transform(bgr_image, m)

"""
Preprocessing
"""
def segment_sky(bgr_image):
    hsv_image = cv.cvtColor(bgr_image, cv.COLOR_BGR2HSV)

    # Light blue sky zone
    lowerbound = np.array([85, 40, 150])
    upperbound = np.array([108, 255, 255])

    mask = cv.inRange(hsv_image, lowerbound, upperbound)
    
    segmented_image = cv.bitwise_and(bgr_image, bgr_image, mask=mask)
    BIN_THRESHOLD = 0.5

    gray_image = cv.cvtColor(segmented_image, cv.COLOR_BGR2GRAY)

    threshold = int(255 * BIN_THRESHOLD)
    ret_thresh, gray_image = cv.threshold(gray_image, threshold, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    return gray_image

def attenuate_sky_light(bgr_image, sky_threshold = 0.3):
    sky_image = segment_sky(bgr_image)
    sky_pixels = sky_image[sky_image > 127].size
    total_pixels = sky_image.size
    sky_ratio = float(sky_pixels) / total_pixels

    if sky_ratio > sky_threshold:
        B, G, R = cv.split(bgr_image)
        return cv.merge([cv.equalizeHist(B), cv.equalizeHist(G), cv.equalizeHist(R)])
    else:
        return bgr_image

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
    alpha = 255 / (maximum_gray - minimum_gray + 1e-6)
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
    return cv.pyrMeanShiftFiltering(bgr_image, 10, 25, 100)

def binarize(gray_image, threshold_percent = 0.75, return_thresh = False):
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
    upperbound_1 = np.array([15, 255, 255])
    # Second zone of reds
    lowerbound_2 = np.array([135, 40, 40])
    upperbound_2 = np.array([180, 255, 255])

    red_1 = cv.inRange(hsv_image, lowerbound_1, upperbound_1)
    red_2 = cv.inRange(hsv_image, lowerbound_2, upperbound_2)

    if return_split:
        return red_1, red_2
    
    return cv.bitwise_or(red_1, red_2)

def extract_blue_hsv(hsv_image):
    lowerbound = np.array([100, 40, 70])
    upperbound = np.array([140, 255, 255])

    return cv.inRange(hsv_image, lowerbound, upperbound)

def extract_white_hsv(hsv_image):
    lowerbound = np.array([0, 0, 127])
    upperbound = np.array([255, 80, 255])

    return cv.inRange(hsv_image, lowerbound, upperbound)

def segment_reds(bgr_image):
    smooth_image = cv.edgePreservingFilter(bgr_image, flags=cv.NORMCONV_FILTER, sigma_s=50, sigma_r=0.5)

    hsv_image = cv.cvtColor(smooth_image, cv.COLOR_BGR2HSV)

    mask = extract_red_hsv(hsv_image=hsv_image)

    segmented_image = cv.bitwise_and(smooth_image, smooth_image, mask=mask)

    gray_image = weighted_gray_transform(segmented_image, [0, 0, 1])

    gray_image = binarize(gray_image=gray_image, threshold_percent=0.5)

    return gray_image

def segment_blues(bgr_image):
    smooth_image = cv.edgePreservingFilter(bgr_image, flags=cv.NORMCONV_FILTER, sigma_s=50, sigma_r=0.5)

    hsv_image = cv.cvtColor(smooth_image, cv.COLOR_BGR2HSV)

    mask = extract_blue_hsv(hsv_image=hsv_image)

    segmented_image = cv.bitwise_and(smooth_image, smooth_image, mask=mask)

    gray_image = weighted_gray_transform(segmented_image, [1, 0, 0])

    gray_image = binarize(gray_image=gray_image, threshold_percent=0.5)

    return gray_image

def segment(bgr_image):
    smooth_image = cv.edgePreservingFilter(bgr_image, flags=cv.NORMCONV_FILTER, sigma_s=50, sigma_r=0.5)
    B, G, R = cv.split(smooth_image)
    hsv_image = cv.cvtColor(smooth_image, cv.COLOR_BGR2HSV)    
    _, S, _ = cv.split(hsv_image)

    B, G, R = np.float64(B), np.float64(G), np.float64(R)
    S = np.float64(S)

    M, N, _ = smooth_image.shape
    hd_blue = np.zeros(shape=(M, N), dtype=np.float64)
    hd_red = np.zeros(shape=(M, N), dtype=np.float64)
    sd = np.zeros(shape=(M, N), dtype=np.float64)
    
    RED_TH = 40
    BLUE_TH = 40
    for i in range(M):
        for j in range(N):
            b, g, r = B[i, j], G[i, j], R[i, j]
            sat = S[i, j]
            max_channel = np.argmax([b, g, r])
            maxI = np.max([b, g, r])
            minI = np.min([b, g, r])

            if max_channel == 0: # B
                hd_blue[i, j] = 1.0 - np.abs(r - g) / (maxI - minI + 1e-6) if (maxI - minI) > BLUE_TH else 0
                hd_red[i, j] = 0
            elif max_channel == 2: # R
                hd_red[i, j] = 1.0 - np.abs(g - b) / (maxI - minI + 1e-6) if (maxI - minI) > RED_TH else 0
                hd_blue[i, j] = 0
            else:
                hd_blue[i, j] = 0
                hd_red[i, j] = 0
            sd[i, j] = sat / 255.0

    hs_red = np.uint8(hd_red * sd * 255)
    hs_blue = np.uint8(hd_blue * sd * 255)

    hs_red = binarize(gray_image=hs_red, threshold_percent=0.5)
    hs_blue = binarize(gray_image=hs_blue, threshold_percent=0.5)

    kernel = cv.getStructuringElement(shape=cv.MORPH_ELLIPSE, ksize=(3, 3))
    hs_red = cv.morphologyEx(hs_red, cv.MORPH_ERODE, kernel=kernel)

    return hs_red, hs_blue

"""
RoI Extraction
"""
def getContours(image, return_hierarchy=False):
    contours, hierarchy = cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    
    if return_hierarchy:
        return contours, hierarchy
    return contours

def getConvexHulls(contours):
    return [cv.convexHull(contour) for contour in contours]

def edge_detection(gray_image):
    # Apply morphological operation to soften possible artefacts
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, ksize=(3, 3))
    morph_image = cv.morphologyEx(gray_image, cv.MORPH_CLOSE, kernel, iterations=1)

    return cv.Canny(morph_image, threshold1=100, threshold2=200, apertureSize=5)

def mergeROI(rois):
    # Sort RoI by X-coordinate and width to merge RoI
    sorted_rois = sorted(rois, key=lambda roi: (roi[0][0], -roi[0][2], roi[0][1], -roi[0][3]))

    i = 0
    while True:
        if i >= len(sorted_rois):
            break

        pivot = sorted_rois[i]
        pivot_x0, pivot_y0, pivot_x1, pivot_y1 = pivot[0][0], pivot[0][1], pivot[0][0] + pivot[0][2], pivot[0][1] + pivot[0][3]
        j = i + 1
        while True:
            if j >= len(sorted_rois):
                break

            other = sorted_rois[j]
            other_x0, other_y0, other_x1, other_y1 = other[0][0], other[0][1], other[0][0] + other[0][2], other[0][1] + other[0][3]
        
            # Beginning of other RoI is already past the ending of the pivot RoI
            if other_x0 > pivot_x1:
                break

            # Check if it's inside, and delete if so, otherwise advance
            if other_y0 > pivot_y0 and other_y1 < pivot_y1 and other_x1 < pivot_x1:
                sorted_rois.pop(j)
            else:
                j += 1
        
        i += 1
    return sorted_rois

def extractROI(edge_image, red_image, blue_image, roi_type):
    kernel = cv.getStructuringElement(shape=cv.MORPH_ELLIPSE, ksize=(3, 3))
    morph_image = cv.morphologyEx(edge_image, cv.MORPH_DILATE, kernel, iterations=1)
    contours = getContours(morph_image)

    # Apply convex hulls to close off shapes
    contours = getConvexHulls(contours)

    # Approximate contour
    contours = [cv.approxPolyDP(contour, 0.004 * cv.arcLength(contour, True), True) for contour in contours]

    rois = [(cv.boundingRect(contour), contour) for contour in contours]


    i = 0
    while True:
        if i >= len(rois):
            break

        (x, y, w, h), contours = rois[i]
        roi_size = w * h

        SIZE_THRESHOLD = 15 * 15
        if roi_size < SIZE_THRESHOLD:
            rois.pop(i)
            continue

        BIG_SIGN = 50 * 50
        aspect_ratio = float(w) / (h + 1e-6)
        if roi_type == "red":
            if roi_size < BIG_SIGN:
                ASPECT_RATIO_MIN = 0.7
                ASPECT_RATIO_MAX = 1.3
                if aspect_ratio < ASPECT_RATIO_MIN or aspect_ratio > ASPECT_RATIO_MAX: 
                    rois.pop(i)
                    continue
            else:
                ASPECT_RATIO_MIN = 0.45
                ASPECT_RATIO_MAX = 1.55
                if aspect_ratio < ASPECT_RATIO_MIN or aspect_ratio > ASPECT_RATIO_MAX: 
                    rois.pop(i)
                    continue
        elif roi_type == "blue":
            if roi_size < BIG_SIGN:
                ASPECT_RATIO_MIN = 0.7
                ASPECT_RATIO_MAX = 1.5
                if aspect_ratio < ASPECT_RATIO_MIN or aspect_ratio > ASPECT_RATIO_MAX: 
                    rois.pop(i)
                    continue
            else:
                ASPECT_RATIO_MIN = 0.45
                ASPECT_RATIO_MAX = 2.0
                if aspect_ratio < ASPECT_RATIO_MIN or aspect_ratio > ASPECT_RATIO_MAX: 
                    rois.pop(i)
                    continue

        blue_pixels = 0
        red_pixels = 0
        contourArea = 0
        for xi in range(x, x+w):
            for yi in range(y, y+h):
                if cv.pointPolygonTest(contours, (xi, yi), measureDist=False) >= 0:
                    if red_image[yi, xi] > 127:
                        red_pixels += 1
                    if blue_image[yi, xi] > 127:
                        blue_pixels += 1
                    contourArea += 1

        blue_ratio = blue_pixels / (roi_size + 1e-6)
        red_ratio = red_pixels / (roi_size + 1e-6)
        red_blue_ratio = red_pixels / (blue_pixels + 1e-6)

        if roi_type == "red":
            if red_ratio < 0.10:
                rois.pop(i)
                continue
        elif roi_type == "blue":
            if blue_ratio < 0.4:
                rois.pop(i)
                continue
            pass
        rois[i] = ((x, y, w, h), contours, red_ratio, blue_ratio, red_blue_ratio, contourArea)

        i += 1
        
    rois = mergeROI(rois)
    return rois

"""
Shape detection
"""
def corner_detection(roi):
    (x, y, w, h), contours, _, _, _, _ = roi
    region = np.zeros(shape=(h, w), dtype=np.uint8)
    cv.drawContours(region, [contours], 0, color=(255), offset=(-x, -y))
    region_32f = np.float32(region)

    corners = cv.cornerHarris(region_32f, 6, 3, 0.04)
    corners = cv.dilate(corners, None)

    norm_corners = np.empty(corners.shape, dtype=np.float32)
    cv.normalize(corners, norm_corners, 255.0, 0.0, cv.NORM_INF)
    norm_corners = cv.convertScaleAbs(norm_corners)

    CORNER_THRESHOLD = 60

    ND = 5
    P = 1.0 / ND
    dw, dh = int(P * w), int(P * h)

    tl = 0.25 * (norm_corners[0:dh, 0:dw].max() > CORNER_THRESHOLD)
    tc = 0.25 * (norm_corners[0:dh, (ND // 2)*dw:(ND // 2 + 1)*dw].max() > CORNER_THRESHOLD)
    tr = 0.25 * (norm_corners[0:dh, (ND - 1)*dw:].max() > CORNER_THRESHOLD)

    ml = 0.25 * (norm_corners[(ND // 2)*dh:(ND // 2 + 1)*dh, 0:dw].max() > CORNER_THRESHOLD)
    mr = 0.25 * (norm_corners[(ND // 2)*dh:(ND // 2 + 1)*dh, (ND-1)*dw:].max() > CORNER_THRESHOLD)

    bl = 0.25 * (norm_corners[(ND-1)*dh:, 0:dw].max() > CORNER_THRESHOLD)
    bc = 0.25 * (norm_corners[(ND-1)*dh:, (ND // 2)*dw:(ND // 2 + 1)*dw].max() > CORNER_THRESHOLD)
    br = 0.25 * (norm_corners[(ND-1)*dh:, (ND-1)*dw:].max() > CORNER_THRESHOLD)

    sqp = max([
        clamp(tl + tr + br + bl - 0.5 * (ml + mr + tc + bc), 0.0, 1.0),
        clamp(0.65 * (bl + br + tr + tc) - tl - 0.5 * ml - 0.8 * (bc + mr), 0.0, 1.0), # oriented rightwards up
        clamp(0.65 * (bl + br + tl + tc) - tr - 0.5 * mr - 0.8 * (bc + ml), 0.0, 1.0), # oriented leftwards up
        clamp(0.65 * (tl + tr + bl + bc) - br - 0.5 * mr - 0.8 * (tc + ml), 0.0, 1.0), # oriented rightwards down
        clamp(0.65 * (tl + tr + br + bc) - bl - 0.5 * ml - 0.8 * (tc + mr), 0.0, 1.0), # oriented leftwards down
    ])
    tup = clamp(1/0.75 * (bl + br + tc) - 2.0 * (tl + tr) - 0.9 * (ml + mr), 0.0, 1.0)
    tdp = clamp(1/0.75 * (tl + tr + bc) - 1.5 * (bl + br) - 0.9 * (ml + mr), 0.0, 1.0)
    circle = clamp(tc + bc + ml + mr -0.25 * (tl + tr + bl + br), 0.0, 1.0)

    return sqp, max(tup, tdp), circle

def extractObjects(bgr_image, contours):
    out_image = np.zeros_like(bgr_image)

    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)
        out_image[y:y+h, x:x+w] = bgr_image[y:y+h, x:x+w]
    
    return out_image