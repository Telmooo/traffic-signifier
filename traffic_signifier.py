import cv2 as cv
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from skimage.exposure import is_low_contrast
import math

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm, colors

def handleMouseEvent(event, x, y, flags, param):
        
    if event == cv.EVENT_LBUTTONUP :
        pixel = param[y, x]
        mouse_pressed = True
        print(f"({pixel[0]}, {pixel[1]}, {pixel[2]})")

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
    cv.namedWindow(title)
    cv.setMouseCallback(title, handleMouseEvent, img)
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
        epsilon = 0.01 * cv.arcLength(contour, True)
        a = cv.approxPolyDP(contour, epsilon, True)
        
        approximations.append(a)
    return approximations
        
    
    


def get_convex_hulls(contours):
    # find convex hull object for each contour
    hulls = []
    for contour in contours:
        hull = cv.convexHull(contour)
        hulls.append(hull)
    
    # TODO: may want to approximate
    # approximate_contours(drawing, hu)
    return hulls
    

def hue_to_rgb(hue):
    s = 1
    v = 1

    c = v * s
    x = c * (1 -  divmod(hue / 60, 2)[0])
    m = v - c

    if hue < 60:
        (r, g, b) = (c, x, 0)
    elif hue < 120:
        (r, g, b) = (x, c, 0)
    elif hue < 180:
        (r, g, b) = (0, c, x)
    elif hue < 240:
        (r, g, b) = (0, x, c)
    elif hue < 300:
        (r, g, b) = (x, 0, c)
    else:
        (r, g, b) = (c, 0, x)

    (r, g, b) = (
        (r + m) * 255,
        (g + m) * 255,
        (b + m) * 255,
    )

    return (r, g, b)
    
 
def find_contours(img):
    ADAPT_THRESH = 51
    # https://docs.opencv.org/3.4/d7/d1d/tutorial_hull.html
    

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    # ret, thresh = cv.threshold(gray, 127, 255, 0)
    # thresh_img = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, ADAPT_THRESH, 1)
    
    _, thresh_img = cv.threshold(gray, 200, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)    
    
    
    contours, _hierarchy = cv.findContours(thresh_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE )
    
    # contours = approximate_contours(contours)
    
    # contours = sorted(contours, key=lambda x : -cv.contourArea(x))[:10]
    
   
    
    # drawing = np.zeros((thresh_img.shape[0], thresh_img.shape[1], 3), dtype=np.uint8)
    # for c in contours:
        
    #     if len(c) == 3:
    #         print("Triangle")
    #         cv.drawContours(drawing, [c], 0, (0,255,255))
    #     elif len(c) == 4:
    #         print("Square")
    #         cv.drawContours(drawing, [c], 0, (255, 0, 0), 1)
    #     elif len(c) == 8:
    #         print("Octagon")
    #         cv.drawContours(drawing, [c], 0, (0,0,255))
    #     else:
    #         print("Other")
    #         cv.drawContours(drawing, [c], 0, (0,255,0))
            
    # show_img(drawing)
    # hulls = get_convex_hulls(contours)
    
    # boundingRect = []
    # for approx_c in contours:
    #     boundingRect.append()
        
        
    # # draw contours + hull results 
    drawing = np.zeros((thresh_img.shape[0], thresh_img.shape[1], 3), dtype=np.uint8)
    for i, c in enumerate(contours):
        b_rect = cv.boundingRect(c)
        
        b_rect_area = b_rect[3] * b_rect[2] # bounding rect (normal)
        
        # b_rect = cv.minAreaRect(c)
        # b_rect_area = abs(b_rect[0][0]-b_rect[1][0]) * abs(b_rect[0][1]-b_rect[1][1])
        
        c_area = cv.contourArea(c)
        ratio = c_area/b_rect_area
        
        cv.rectangle(drawing, (int(b_rect[0]), int(b_rect[1])), \
            (int(b_rect[0]+b_rect[2]), int(b_rect[1]+b_rect[3])), (255, 255, 255), 2)
        
        if ratio >= 0.90:
            print("Square")
            cv.drawContours(drawing, [c], 0, (255, 0, 0))
        elif ratio >= 0.70:
            (x,y),radius = cv.minEnclosingCircle(c)
            center = (int(x),int(y))
            radius = int(radius)
            cv.circle(img,center,radius,(0,255,0),2)
            
            circ_area = math.pi * radius**2
            
            if c_area >= 0.95:
                print("Circle")
                cv.drawContours(drawing, [c], 0, (0,255,0))
            elif c_area >= 0.75:
                print("Octagon")
                cv.drawContours(drawing, [c], 0, (0,0,255))
            
        elif ratio >= 0.35:
            print("Triangle")
            cv.drawContours(drawing, [c], 0, (0,255,255))
        else:
            cv.drawContours(drawing, [c], 0, (255,255,255))
            
    show_img(drawing)   



def image_red_ratio(hsv):
    mask = cv.inRange(hsv, np.array([0, 0, 25]), np.array([10, 255, 255]))
    mask2 = cv.inRange(hsv, np.array([135, 0, 25]), np.array([179, 255, 255]))
    
    mask = cv.bitwise_or(mask, mask2)
    ratio = cv.countNonZero(mask) / hsv.size/3
    return ratio, mask

def image_blue_ratio(hsv):
    lower_blue = np.array([100, 160, 40])
    upper_blue = np.array([120, 255, 255])
    mask = cv.inRange(hsv, lower_blue, upper_blue)
    ratio = cv.countNonZero(mask) / hsv.size/3
    return ratio, mask

if __name__ == '__main__':
    # CONTRAST_THRESH = 0.7
    
    # TODO morphology on blue square with crosswalks
    
    dataDir = Path("./data/images")
    
    signDir = Path("./data/signs")

    # img_path = str(dataDir / "road55.png")
    img_path = str(dataDir / "road875.png")
    # img_path = str(dataDir / "road369.png") # TODO: Hell on Earth
    # img_path = str(signDir / "warning/warning-crossroad-stop.png")
    img = cv.imread(img_path)
    
    
    if (is_low_contrast(img, 0.5)):
        print("Low contrast: " + img_path)
        img = clahe_equalization(img)

    segmented_img = cv.pyrMeanShiftFiltering(img, 10, 15)
    show_img(img, "Original")
    show_img(segmented_img, "Mean Shift")

    # _regions, boxes = mser(img)
    
    # for box in boxes:
    #     x, y, w, h = box
    #     region_img = img[y:y+h, x:x+w]

    # kernel = np.ones((1,1),np.uint8)
    # morphed_img = cv.morphologyEx(segmented_img, cv.MORPH_CLOSE, kernel)



    hsv = cv.cvtColor(segmented_img, cv.COLOR_BGR2HSV)
    red_ratio, red_mask = image_red_ratio(hsv)
    blue_ratio, blue_mask = image_blue_ratio(hsv)
        
    reds = cv.bitwise_and(img, img, mask = red_mask)
    blues = cv.bitwise_and(img, img, mask = blue_mask)
    
    show_img(reds)
    show_img(blues)
    
    find_contours(reds)
    
    find_contours(blues)





