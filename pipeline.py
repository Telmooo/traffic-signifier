import traffic_signifier as ts

import cv2 as cv
import numpy as np
from pathlib import Path


def test_img(img_path: str):
    img = cv.imread(img_path)
    
    # if (is_low_contrast(img, 0.5)):
    print("Low contrast: " + img_path)
    img = ts.clahe_equalization(img)


    # segmented_img = cv.pyrMeanShiftFiltering(img, 10, 25, 100)
    segmented_img = cv.pyrMeanShiftFiltering(img, 10, 15, 100)

    hsv = cv.cvtColor(segmented_img, cv.COLOR_BGR2HSV)
    red_ratio, red_mask = ts.image_red_ratio(hsv)
    blue_ratio, blue_mask = ts.image_blue_ratio(hsv)
        
    reds = cv.bitwise_and(img, img, mask = red_mask)
    blues = cv.bitwise_and(img, img, mask = blue_mask)
    
    red_contours = ts.find_contours(reds)
    blue_contours = ts.find_contours(blues)

    # drawing = img.copy()
    drawing = np.zeros_like(img)

    ts.draw_contours_boxes(drawing, red_contours)
    ts.draw_contours_boxes(drawing, blue_contours)

    ts.draw_contours(drawing, red_contours)
    ts.draw_contours(drawing, blue_contours)

    ts.show_img(drawing)




if __name__ == '__main__':
    dataDir = Path("./data/images")
    signDir = Path("./data/signs")

    test_img(str(dataDir / "road873.png"))
