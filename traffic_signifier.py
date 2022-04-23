import cv2 as cv
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from skimage.exposure import is_low_contrast

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm, colors



def display_color_hist(img):
    """Display color histogram

    Args:
        img : Image 
    """
    color = ('b','g','r')
    for i,col in enumerate(color):
        histr = cv.calcHist([img],[i],None,[256],[0,256])
        plt.plot(histr,color = col)
        plt.xlim([0,256])
    plt.show()
    
def show_img(img, title="Image"):
    cv.imshow(title, img)
    cv.waitKey(0)
    cv.destroyWindow(title)

def clahe_equalization(lowContrast):
    """Perform CLAHE Equalization

    Args:
        lowContrast (Any): low constrast image

    Returns:
        Any: Equalized image
    """
    hsv = cv.cvtColor(lowContrast, cv.COLOR_BGR2HSV)
    (h,s,v) = cv.split(hsv)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe_s = clahe.apply(s)
    clahe_v = clahe.apply(v)

    clahe_img = cv.merge([h, clahe_s, clahe_v])

    # convert before display
    final_clahe = cv.cvtColor(clahe_img, cv.COLOR_HSV2BGR)
    
    return final_clahe

def hist_equalization(lowContrast):
    """Perform Histogram Equalization

    Args:
        lowContrast (img): low contrast image
        
    Return:
        img: equalized image
    """
    hsv = cv.cvtColor(lowContrast, cv.COLOR_BGR2HSV)
    (h,s,v) = cv.split(hsv)
    
    s = cv.equalizeHist(s)
    v = cv.equalizeHist(v)
    
    final_he = cv.merge([h,s,v])
    final_he = cv.cvtColor(final_he, cv.COLOR_HSV2BGR)
    
    return final_he
    

def rgb_3d_plot(rgb_img):
    (r,g,b) = cv.split(rgb_img)
    
    fig = plt.figure()
    axis = fig.add_subplot(1,1,1, projection="3d")
    
    pixel_colors = rgb_img.reshape((np.shape(rgb_img)[0]*np.shape(rgb_img)[1], 3))
    norm = colors.Normalize(vmin=-1.,vmax=1.)
    norm.autoscale(pixel_colors)
    pixel_colors = norm(pixel_colors).tolist()
    
    axis.scatter(r.flatten(), g.flatten(), b.flatten(), facecolors=pixel_colors, marker=".")
    axis.set_xlabel("Red")
    axis.set_ylabel("Green")
    axis.set_zlabel("Blue")
    
    plt.show()
    
def hsv_3d_plot(hsv_img):
    (h,s,v) = cv.split(hsv_img)
        
    fig = plt.figure()
    axis = fig.add_subplot(1,1,1, projection="3d")
    
    pixel_colors = hsv_img.reshape((np.shape(hsv_img)[0]*np.shape(hsv_img)[1], 3))
    norm = colors.Normalize(vmin=-1.,vmax=1.)
    norm.autoscale(pixel_colors)
    pixel_colors = norm(pixel_colors).tolist()
    
    axis.scatter(h.flatten(), s.flatten(), v.flatten(), facecolors=pixel_colors, marker=".")
    axis.set_xlabel("Hue")
    axis.set_ylabel("Saturation")
    axis.set_zlabel("Value")
    
    plt.show()
    

def mser(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    mser = cv.MSER_create()
    regions, boxes = mser.detectRegions(gray)
    
    return regions, boxes
    
def flann_matcher(des1, des2, img1, img2, kp1, kp2):
    matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED)
    matches = matcher.knnMatch(des1, des2, k=2)
    
    ratio_thresh = 0.7
    good_matches = []
    # ratio test as per Lowe's paper
    for m,n in matches:
        if m.distance < ratio_thresh*n.distance:
            good_matches.append(m)
    
    img_matches = np.empty((max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1], 3), dtype=np.uint8)
    
    cv.drawMatches(img1, kp1, img2, kp2, good_matches, img_matches, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(img_matches)
    plt.axis('off')
    plt.title("FLANN")
    plt.show()
    
def approximate_contours(contours):
    # TODO ver se shape é circulo, qaudrado ou retângulo se n for excluir
    approximations = []
    for contour in contours:
        epsilon = 0.02 * cv.arcLength(contour, True)
        a = cv.approxPolyDP(contour, epsilon, True)
        
        approximations.append(a)
    return approximations
        
    
    
def threshold_segmentation(gray, thresh):
    _, thresh_img = cv.threshold(gray, 200, thresh, cv.THRESH_BINARY+cv.THRESH_OTSU)
    show_img(thresh_img)
    return thresh_img

def get_convex_hulls(contours):
    # find convex hull object for each contour
    hulls = []
    for contour in contours:
        hull = cv.convexHull(contour)
        hulls.append(hull)
    
    # TODO: may want to approximate
    # approximate_contours(drawing, hu)
    return hulls
    
    
 
def find_contours(img):
    ADAPT_THRESH = 51
    # https://docs.opencv.org/3.4/d7/d1d/tutorial_hull.html
    

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    # ret, thresh = cv.threshold(gray, 127, 255, 0)
    thresh = 255
    # thresh_img = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, ADAPT_THRESH, 1)
    
    thresh_img = threshold_segmentation(gray, thresh)
    
    # canny_img = cv.Canny(gray, thresh, thresh*2)
    
    contours, _hierarchy = cv.findContours(thresh_img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    # approx_contours = approximate_contours(contours)
      
    
    boundingRect = []
    final_approx = []
    for approx_c in contours:
        b_rect = cv.boundingRect(approx_c)
        
        b_area = abs(b_rect[0] - b_rect[2]) * abs(b_rect[1] - b_rect[3])
        
        if (b_area > 0):
            extent = cv.contourArea(approx_c) / b_area
        else:
            continue
        
        if (extent > 0.75): 
            boundingRect.append(b_rect)
            final_approx.append(approx_c)
       
    # draw contours + hull results 
    drawing = np.zeros((thresh_img.shape[0], thresh_img.shape[1], 3), dtype=np.uint8)
    for i in range(len(final_approx)):
        cv.drawContours(drawing, final_approx, i, (0, 0, 255))
        cv.rectangle(drawing, (int(boundingRect[i][0]), int(boundingRect[i][1])), \
        (int(boundingRect[i][0]+boundingRect[i][2]), int(boundingRect[i][1]+boundingRect[i][3])), (0, 255, 0), 2)
        
    
    
    cv.imshow("Countours", drawing)
    cv.waitKey(0)
    cv.destroyAllWindows()
    

def image_red_ratio(hsv):
    mask = cv.inRange(hsv, np.array([0, 40, 25]), np.array([5, 255, 255]))
    mask2 = cv.inRange(hsv, np.array([170, 40, 25]), np.array([179, 255, 255]))
    
    mask = cv.bitwise_or(mask, mask2)
    ratio = cv.countNonZero(mask) / hsv.size/3
    return ratio, mask

def image_blue_ratio(hsv):
    lower_blue = np.array([100, 105, 40])
    upper_blue = np.array([120, 255, 255])
    mask = cv.inRange(hsv, lower_blue, upper_blue)
    ratio = cv.countNonZero(mask) / hsv.size/3
    return ratio, mask

if __name__ == '__main__':
    # CONTRAST_THRESH = 0.7
    
    dataDir = Path("./data/images")

    img_path = str(dataDir / "road825.png")
    img = cv.imread(img_path)
    # gray = cv.imread(img_path, cv.IMREAD_GRAYSCALE)

    # display_color_hist(img)
    # show_img(img, "original")
    
    # # hsv  = cv.cvtColor(img, cv.COLOR_BGR2HSV_FULL)
    # # hsv_3d_plot(hsv)
    # if (is_low_contrast(gray, CONTRAST_THRESH)):
    #     print("Low Constrast Image")
        
    #     clahe_img = clahe_equalization(img)
    #     show_img(clahe_img, "CLAHE")
        
    #     hist_img = hist_equalization(img)
    #     show_img(hist_img, "HIST")
    
    # mser()
    
    _regions, boxes = mser(img)
    
    # for box in boxes:
    #     x, y, w, h = box
    #     region_img = img[y:y+h, x:x+w]
        
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    red_ratio, red_mask = image_red_ratio(hsv)
    blue_ratio, blue_mask = image_blue_ratio(hsv)
    
    mask = cv.bitwise_or(red_mask, blue_mask)
    
    print(red_mask.shape, blue_mask.shape, mask.shape)
    img_ = cv.bitwise_and(img, img, mask = mask)
    
    show_img(img_)
    
    find_contours(img_)