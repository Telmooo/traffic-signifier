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
    

def mser():
    img = cv.imread(str(Path("./data/images/road57.png")))
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    mser = cv.MSER_create()
    _regions, boxes = mser.detectRegions(gray)
    for box in boxes:
        x,y,w,h = box
        cv.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
    
    cv.imshow("MSER", img)
    cv.waitKey(0)
    
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

if __name__ == '__main__':
    # CONTRAST_THRESH = 0.7
    
    # dataDir = Path("./data/images")

    # img_path = str(dataDir / "road153.png")
    # img = cv.imread(img_path)
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
    
    mser()