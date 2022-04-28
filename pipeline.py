import utils.image_processing as imp

import os

import cv2 as cv
import numpy as np

def handleMouseEvent(event, x, y, flags, param):
        
    if event == cv.EVENT_LBUTTONUP :
        pixel = param[y, x]
        mouse_pressed = True
        print(f"({pixel[0]}, {pixel[1]}, {pixel[2]})")
def show_img(img, title:str="Image"):
    cv.namedWindow(title)
    cv.setMouseCallback(title, handleMouseEvent, img)
    cv.imshow(title, img)
    cv.waitKey(0)
    cv.destroyWindow(title)

def preprocess_image(bgr_image):
    out_image = imp.hsv_clahe_equalization(
        bgr_image=bgr_image,
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )
    out_image = imp.automatic_brightness_contrast(
        bgr_image=out_image,
        clip_hist_percent=0.05,
        use_scale_abs=True
    )

    out_image = imp.meanShiftFiltering(out_image)
    

    return out_image

def segment_reds(bgr_image):
    hsv_image = cv.cvtColor(bgr_image, cv.COLOR_BGR2HSV)
    mask = imp.extract_red_hsv(hsv_image=hsv_image)
    out_image = cv.bitwise_and(bgr_image, bgr_image, mask=mask)
    out_image = imp.binarize(out_image, threshold_percent=0.75)
    out_image = imp.removeSmallComponents(out_image, size_threshold=200)
    return out_image

def segment_blues(bgr_image):
    hsv_image = cv.cvtColor(bgr_image, cv.COLOR_BGR2HSV)
    mask = imp.extract_blue_hsv(hsv_image=hsv_image)
    out_image = cv.bitwise_and(bgr_image, bgr_image, mask=mask)
    out_image = imp.binarize(out_image, threshold_percent=0.75)
    out_image = imp.removeSmallComponents(out_image, size_threshold=200)
    return out_image

def extract_ROI(original_image, segmented_image):

    contours = imp.getContours(segmented_image)

    test_image = np.zeros_like(segmented_image)
    for i in range(len(contours)):
        cv.drawContours(test_image, contours, i, (255))

    show_img(test_image)

    out_image = imp.extractObjects(original_image, contours)

    return out_image

def segment_red_ROI(bgr_image):
    hsv_image = cv.cvtColor(bgr_image, cv.COLOR_BGR2HSV)
    red_mask = imp.extract_red_hsv(hsv_image=hsv_image)
    white_mask = imp.extract_white_hsv(hsv_image=hsv_image)
    mask = cv.bitwise_or(red_mask, white_mask)
    out_image = cv.bitwise_and(bgr_image, bgr_image, mask=mask)
    out_image = imp.binarize(out_image, threshold_percent=0.75)
    out_image = imp.removeSmallComponents(out_image, size_threshold=200)
    return out_image

def segment_blue_ROI(bgr_image):
    hsv_image = cv.cvtColor(bgr_image, cv.COLOR_BGR2HSV)
    blue_mask = imp.extract_blue_hsv(hsv_image=hsv_image)
    white_mask = imp.extract_white_hsv(hsv_image=hsv_image)
    mask = cv.bitwise_or(blue_mask, white_mask)
    out_image = cv.bitwise_and(bgr_image, bgr_image, mask=mask)
    out_image = imp.binarize(out_image, threshold_percent=0.75)
    out_image = imp.removeSmallComponents(out_image, size_threshold=200)
    return out_image

circularity_ratios = {
    "circle": 1,
    "quadrilateral": 0.70,
    "octagon": 0.92,
    "triangle": 0.41,
    # "diamond": 0.64,
}
circularity_ratios["other"] = 0.5 # np.mean(list(circularity_ratios.values()))

extent_ratios = {
    "circle": 0.785,
    "quadrilateral": 1,
    "octagon": 0.829,
    "triangle": 0.498,
    # "diamond": 0.5,
}
extent_ratios["other"] = 0.5 # np.mean(list(extent_ratios.values()))

minextent_ratios = {
    "circle": 0.785,
    "quadrilateral": 1,
    "octagon": 0.829,
    "triangle": 0.498,
    # "diamond": 1
}
minextent_ratios["other"] = 0.5 # np.mean(list(minextent_ratios.values()))

def get_shape(contour, return_probabilities = False):
    (_brect_x, _brect_y, brect_w, brect_h) = cv.boundingRect(contour)
    brect_area = brect_w * brect_h

    (minrect_x0, minrect_y0), (minrect_x1, minrect_y1), _minrect_angle = cv.minAreaRect(contour)
    minrect_area = abs(minrect_x0 - minrect_x1) * abs(minrect_y0 - minrect_y1)

    (_circle_x, _circle_y), circle_radius = cv.minEnclosingCircle(contour)
    circle_area = np.pi * circle_radius * circle_radius

    # mintriangle_area, _triangle = cv.minEnclosingTriangle(contour)

    contourArea = cv.contourArea(contour)
    contourPerimeter = cv.arcLength(contour, True)
    # Measures how compact the shape is
    circularity = (4 * np.pi * contourArea) / (contourPerimeter * contourPerimeter + 1e-4)

    extent = contourArea / brect_area
    min_extent = contourArea / minrect_area
    
    circle_ratio = contourArea / circle_area
    # triangle_ratio = contourArea / mintriangle_area

    metrics = ["circularity", "circle_ratio", "extent", "min_extent"]
    ratios = [circularity, circle_ratio, extent, min_extent]
    ratio_tables = [circularity_ratios, circularity_ratios, extent_ratios, minextent_ratios]
    n_metrics = len(ratios)

    classes = ["circle", "quadrilateral", "octagon", "triangle", "other"] # Add diamond
    n_classes = len(classes)

    probability_table = np.zeros(shape=(n_metrics + 1, n_classes + 1))
    for i in range(n_metrics):
        ratio = ratios[i]
        table = ratio_tables[i]
        for j in range(n_classes):
            shape_class = classes[j]
            class_ratio = table[shape_class]
            probability_table[i, j] = abs(ratio - class_ratio) / class_ratio
        
        probability_table[i, n_classes] = np.sum(probability_table[i, 0:n_classes])
        probability_table[i, 0:n_classes] = (1 - (probability_table[i, 0:n_classes] / probability_table[i, n_classes])) / (n_classes - 1)
        probability_table[i, n_classes] = np.sum(probability_table[i, 0:n_classes])

    probability_table[n_metrics, :n_classes] = np.sum(probability_table[:-1, :-1], axis=0) / (n_classes - 1)
    probability_table[-1, n_classes] = np.sum(probability_table[-1, 0:n_classes])


    print(probability_table)

    chosen_shape = -1
    max_probability = -1
    for i, shape in enumerate(classes):
        p = probability_table[-1, i] 
        if p > max_probability:
            chosen_shape = shape
            max_probability = p

    if return_probabilities:
        return chosen_shape, probability_table

    return chosen_shape

def detect_red_signs(original_image, segmented_image):
    kernel = np.ones(shape=(5, 5), dtype=np.uint8)
    morph_image = cv.morphologyEx(segmented_image, cv.MORPH_CLOSE, kernel, iterations=1)

    edges_image = cv.Canny(morph_image, 100, 200)

    contours = imp.getContours(edges_image)
    hulls = imp.getConvexHulls(contours)

    approx_contours = [cv.approxPolyDP(hull, 0.01 * cv.arcLength(hull, True), True) for hull in hulls]

    out_image = original_image.copy()
    for contour in approx_contours:
        shape = get_shape(contour)
        print(shape)
        if shape == "circle":
            (x,y),radius = cv.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)
            cv.circle(out_image, center, radius, (0, 255, 0), 2)
        elif shape == "quadrilateral":
            # Doesn't exist, in theory
            x,y,w,h = cv.boundingRect(contour)
            cv.rectangle(out_image, (x,y), (x+w,y+h), (255, 0, 0),2)
        elif shape == "triangle":
            # Doesn't exist, in theory
            x,y,w,h = cv.boundingRect(contour)
            cv.rectangle(out_image, (x,y), (x+w,y+h), (0, 255, 255),2)
        elif shape == "octagon":
            # Doesn't exist, in theory
            x,y,w,h = cv.boundingRect(contour)
            cv.rectangle(out_image, (x,y), (x+w,y+h), (0, 0, 255),2)
        elif shape == "other":
            # Doesn't exist, in theory
            x,y,w,h = cv.boundingRect(contour)
            cv.rectangle(out_image, (x,y), (x+w,y+h), (255, 255, 255),2)

    return out_image

def save_images(outDir, output_dict):
    os.makedirs(outDir, exist_ok=True)

    for name, (subDir, image) in output_dict.items():
        directory = f"{outDir}/{subDir}"
        os.makedirs(directory, exist_ok=True)

        path = f"{directory}/{name}.png"
        cv.imwrite(path, image)

def detect_traffic_signs(name: str, image_path : str, outDir : str, save_all : bool):
    image = cv.imread(image_path)
    
    processed_image = preprocess_image(image)

    red_image = segment_reds(processed_image)
    blue_image = segment_blues(processed_image)

    roi_red_image = extract_ROI(processed_image, red_image)
    roi_blue_image = extract_ROI(processed_image, blue_image)

    roi_red_segment = segment_red_ROI(roi_red_image)
    roi_blue_segment = segment_blue_ROI(roi_blue_image)

    out_image = detect_red_signs(image, roi_red_segment)
    show_img(out_image)

    if save_all:
        output = {
            f"{name}_processed": ("processed", processed_image),
            f"{name}_red_segment": ("red_segmentation", cv.cvtColor(red_image, cv.COLOR_GRAY2BGR)),
            f"{name}_blue_segment": ("blue_segmentation", cv.cvtColor(blue_image, cv.COLOR_GRAY2BGR)),
            f"{name}_red_roi": ("red_roi", roi_red_image),
            f"{name}_blue_roi": ("blue_roi", roi_blue_image),
            f"{name}_red_roi_segment": ("red_roi_segmentation", roi_red_segment),
            f"{name}_blue_roi_segment": ("blue_roi_segmentation", roi_blue_segment),
        }
    else:
        output = {

        }

    save_images(outDir, output)

# def test_img(img_path: str):
#     img = cv.imread(img_path)
    
#     # if (is_low_contrast(img, 0.5)):
#     print("Low contrast: " + img_path)
#     # img = ts.clahe_equalization(img)


#     # segmented_img = cv.pyrMeanShiftFiltering(img, 10, 25, 100)
#     segmented_img = cv.pyrMeanShiftFiltering(img, 10, 15, 100)

#     hsv = cv.cvtColor(segmented_img, cv.COLOR_BGR2HSV)
    # red_ratio, red_mask = ts.image_red_ratio(hsv)
    # blue_ratio, blue_mask = ts.image_blue_ratio(hsv)
        
    # reds = cv.bitwise_and(img, img, mask = red_mask)
    # blues = cv.bitwise_and(img, img, mask = blue_mask)
    
    # red_contours = ts.find_contours(reds)
    # blue_contours = ts.find_contours(blues)

    # # drawing = img.copy()
    # drawing = np.zeros_like(img)

    # ts.draw_contours_boxes(drawing, red_contours)
    # ts.draw_contours_boxes(drawing, blue_contours)

    # ts.draw_contours(drawing, red_contours)
    # ts.draw_contours(drawing, blue_contours)

    # ts.show_img(drawing)




if __name__ == '__main__':
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(
        description="Detect traffic signs in the image specified. If input is a directory, detect traffic signs on all images in the folder.",
        add_help=True,
        allow_abbrev=True,
    )

    parser.add_argument(
        "path", help="Path of the image or directory to process."
    )

    parser.add_argument(
        "--out", help="Path of output directory. Defaults to `out`", default="./out"
    )

    parser.add_argument(
        "--annotations", help="Path to annotations for scoring accuracy of detection."
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Controls the verbosity, if enable the program will output more messages and save more images related to the processing."
    )

    args = parser.parse_args()

    path = args.path
    outPath = args.out
    annotationsPath = args.annotations
    
    verbose = args.verbose
    isFile = os.path.isfile(path)
    isDir = os.path.isdir(path)

    if isFile:
        filename, _extension = os.path.splitext(os.path.basename(path))

        detect_traffic_signs(
            name=filename,
            image_path=path,
            outDir=outPath,
            save_all=verbose
        )
    elif isDir:
        pass
    else:
        print(f"Path {path} is unrecognized. Please verify if path corresponds to valid file or directory.", file=sys.stderr)
