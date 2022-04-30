import utils.image_processing as imp
import utils.xml_parser as xp

import os
from collections import defaultdict

import numpy as np
import cv2 as cv
import pandas as pd

def preprocess_image(bgr_image):
    out_image = imp.hsv_clahe_equalization(
        bgr_image=bgr_image,
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )
    out_image = imp.automatic_brightness_contrast(
        bgr_image=out_image,
        clip_hist_percent=0.01,
        use_scale_abs=True
    )

    out_image = imp.meanShiftFiltering(out_image)

    return out_image

# Mathematical ratios
circularity_ratios = {
    "circle": 1,
    "quadrilateral": 0.70,
    "octagon": 0.92,
    "triangle": 0.41,
    # "diamond": 0.64,
}

extent_ratios = {
    "circle": 0.785,
    "quadrilateral": 1,
    "octagon": 0.829,
    "triangle": 0.498,
    # "diamond": 0.5,
}

minextent_ratios = {
    "circle": 0.785,
    "quadrilateral": 1,
    "octagon": 0.829,
    "triangle": 0.498,
    # "diamond": 1
}

red_ratios = {
    "circle": 0.30,
    "quadrilateral": 0.0,
    "octagon": 0.65,
    "triangle": 0.20,
}

blue_ratios = {
    "circle": 0.60,
    "quadrilateral": 0.60,
    "octagon": 0.0,
    "triangle": 0.0,
}

output_classes = defaultdict(lambda: "unknown", {
    ("circle", "red"): "prohibitory",
    ("triangle", "red"): "priority",
    ("octagon", "red"): "stop",

    ("quadrilateral", "blue"): "information",
    ("circle", "blue"): "mandatory",
})

def detect_shape(roi, roi_type, return_probabilities = False):
    (brect_x, brect_y, brect_w, brect_h), contour, red_ratio, blue_ratio, red_blue_ratio, sqp, trg, circle = roi

    contourArea = cv.contourArea(contour)
    contourPerimeter = cv.arcLength(contour, True)
    # Bounding Rectangle Area - used to calculate extent of contour
    brect_area = brect_w * brect_h
    extent = contourArea / brect_area

    # Minimum Bounding Rectangle - takes into account orientation and fits the bounding rectangle thighly
    (min_brect_x0, min_brect_y0), (min_brect_x1, min_brect_y1), min_brect_angle = cv.minAreaRect(contour)
    min_brect_area = abs(min_brect_x0 - min_brect_x1) * abs(min_brect_y0 - min_brect_y1)

    min_extent = contourArea / min_brect_area

    # Circularity - measures how compact the contour is
    circularity = (4 * np.pi * contourArea) / (contourPerimeter * contourPerimeter + 1e-4)

    # Minimum Enclosing Circle - similar to circularity
    (min_circle_x, min_circle_y), circle_radius = cv.minEnclosingCircle(contour)
    min_circle_area = np.pi * circle_radius * circle_radius

    circle_extent = contourArea / min_circle_area

    metrics = ["circularity", "circle_extent", "extent", "min_extent"]
    ratios = [circularity, circle_extent, extent, min_extent]
    ratio_tables = [circularity_ratios, circularity_ratios, extent_ratios, minextent_ratios]
    if roi_type == "red":
        metrics.append("color_ratio")
        ratios.append(red_ratio)
        ratio_tables.append(red_ratios)
    elif roi_type == "blue":
        metrics.append("color_ratio")
        ratios.append(blue_ratio)
        ratio_tables.append(blue_ratios)
    
    metrics.append("corners")
    n_metrics = len(metrics)

    classes = ["circle", "quadrilateral", "octagon", "triangle"] # Add diamond
    n_classes = len(classes)

    probability_table = np.zeros(shape=(n_metrics + 1, n_classes + 1))
    for i in range(n_metrics - 1):
        ratio = ratios[i]
        table = ratio_tables[i]
        for j in range(n_classes):
            shape_class = classes[j]
            class_ratio = table[shape_class]
            probability_table[i, j] = abs(ratio - class_ratio) ** 2 # / class_ratio
        
        probability_table[i, n_classes] = np.sum(probability_table[i, 0:n_classes])
        probability_table[i, 0:n_classes] = (1 - (probability_table[i, 0:n_classes] / probability_table[i, n_classes])) / (n_classes - 1)
        probability_table[i, n_classes] = np.sum(probability_table[i, 0:n_classes])

    # Add corners row
    probability_table[n_metrics - 1, :n_classes] = np.array([circle, sqp, circle, trg])
    probability_table[n_metrics - 1, n_classes] = np.sum(probability_table[n_metrics - 1, 0:n_classes])
    probability_table[n_metrics - 1, :n_classes] /= probability_table[n_metrics - 1, n_classes]

    probability_table[-1, :n_classes] = np.mean(probability_table[:-1, :-1], axis=0)
    probability_table[-1, n_classes] = np.sum(probability_table[-1, 0:n_classes])

    chosen_shape = ""
    max_probability = -1
    for i, shape in enumerate(classes):
        p = probability_table[-1, i] 
        if p > max_probability:
            chosen_shape = shape
            max_probability = p

    CONFIDENCE = 0.1 # Confidence threshold, how much % is needed to obtain majority
    THRESHOLD = (1 / n_classes) * (1 + CONFIDENCE)
    if max_probability < THRESHOLD:
        chosen_shape = "other"
        max_probability = -1

    if roi_type == "red":
        if chosen_shape == "quadrilateral":
            chosen_shape = "other"
        elif chosen_shape == "circle" or chosen_shape == "octagon":
            if red_ratio > 0.45:
                chosen_shape = "octagon"
            else:
                chosen_shape = "circle"
    elif roi_type == "blue":
        if chosen_shape == "triangle" or chosen_shape == "octagon":
            chosen_shape = "other"

    if return_probabilities:
        df = pd.DataFrame(data=probability_table[:, :-1], columns=classes, index=metrics + ["AVG(P)"])
        return chosen_shape, max_probability, df, THRESHOLD

    return chosen_shape

def save_images(outDir, output_dict):
    os.makedirs(outDir, exist_ok=True)

    for name, (subDir, image) in output_dict.items():
        directory = f"{outDir}/{subDir}"
        os.makedirs(directory, exist_ok=True)

        path = f"{directory}/{name}.png"
        cv.imwrite(path, image)

def detect_traffic_signs(name: str, image_path : str, outDir : str, annot_dict : dict, save_all : bool):
    image = cv.imread(image_path)
    
    processed_image = preprocess_image(image)

    red_image = imp.segment_reds(processed_image)
    blue_image = imp.segment_blues(processed_image)

    red_edges = imp.edge_detection(red_image)
    blue_edges = imp.edge_detection(blue_image)

    red_roi = imp.extractROI(red_edges, red_image=red_image, blue_image=blue_image, roi_type="red")
    blue_roi = imp.extractROI(blue_edges, red_image=red_image, blue_image=blue_image, roi_type="blue")

    roi_red_image = np.zeros(shape=(red_edges.shape + (3,)), dtype=np.uint8)
    for roi in red_roi:
        (x, y, w, h), contours, _, _, _ = roi
        cv.rectangle(roi_red_image, (int(x), int(y)), (int(x+w), int(y+h)), (197, 183, 255), 1)
        cv.drawContours(roi_red_image, [contours], 0, color=(255, 255, 255))

    roi_blue_image = np.zeros(shape=(blue_edges.shape + (3,)), dtype=np.uint8)
    for roi in blue_roi:
        (x, y, w, h), contours, _, _, _ = roi
        cv.rectangle(roi_blue_image, (int(x), int(y)), (int(x+w), int(y+h)), (197, 183, 255), 1)
        cv.drawContours(roi_blue_image, [contours], 0, color=(255, 255, 255))

    # Get corner info
    for i in range(len(red_roi)):
        roi = red_roi[i]
        (brect_x, brect_y, brect_w, brect_h), contour, red_ratio, blue_ratio, red_blue_ratio = roi
        sqp, trg, circle = imp.corner_detection(roi, red_edges)
        red_roi[i] = ((brect_x, brect_y, brect_w, brect_h), contour,  red_ratio, blue_ratio, red_blue_ratio, sqp, trg, circle)

    for i in range(len(blue_roi)):
        roi = blue_roi[i]
        (brect_x, brect_y, brect_w, brect_h), contour, red_ratio, blue_ratio, red_blue_ratio = roi
        sqp, trg, circle = imp.corner_detection(roi, blue_edges)
        blue_roi[i] = ((brect_x, brect_y, brect_w, brect_h), contour, red_ratio, blue_ratio, red_blue_ratio, sqp, trg, circle)

    output_image = image.copy()

    for roi in red_roi:
        shape = detect_shape(roi, "red", return_probabilities=False)
        (x, y, w, h), contours, red_ratio, blue_ratio, red_blue_ratio, sqp, trg, circle = roi

        if shape != "other":
            cv.drawContours(output_image, [contours], 0, color=(25, 255, 40), thickness=2)
            output_class = output_classes[(shape, "red")]
            cv.putText(output_image, output_class, (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.6, (25, 255, 40), 2, cv.LINE_AA)

    for roi in blue_roi:
        shape = detect_shape(roi, "blue", return_probabilities=False)
        (x, y, w, h), contours, red_ratio, blue_ratio, red_blue_ratio, sqp, trg, circle = roi
        if shape != "other":
            cv.drawContours(output_image, [contours], 0, color=(25, 255, 40), thickness=2)
            output_class = output_classes[(shape, "blue")]
            cv.putText(output_image, output_class, (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.6, (25, 255, 40), 2, cv.LINE_AA)

    if save_all:
        output = {
            f"{name}_processed": ("processed", processed_image),
            f"{name}_red_segment": ("red_segmentation", cv.cvtColor(red_image, cv.COLOR_GRAY2BGR)),
            f"{name}_blue_segment": ("blue_segmentation", cv.cvtColor(blue_image, cv.COLOR_GRAY2BGR)),
            f"{name}_red_roi": ("red_roi", roi_red_image),
            f"{name}_blue_roi": ("blue_roi", roi_blue_image),
            f"{name}_output": ("output", output_image),
        }
    else:
        output = {
            f"{name}_output": ("output", output_image),
        }

    save_images(outDir, output)

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
        "--annotations", "-a", help="Path to annotations directory for scoring accuracy of detection."
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
    
    annot_dict = {}
    if isFile:
        filename, _extension = os.path.splitext(os.path.basename(path))
        
        if annotationsPath:
            annot_dict = xp.parse(f'{annotationsPath}{filename}.xml')
            
        print(annot_dict)
            
        detect_traffic_signs(
            name=filename,
            image_path=path,
            outDir=outPath,
            save_all=verbose,
            annot_dict=annot_dict
        )
    elif isDir:
        
        if annotationsPath:
            annot_dict = xp.from_dir(annotationsPath)
        print(annot_dict)
        
        dir_files = os.listdir(path)
        for f in os.listdir(path):
            if f.endswith('.png') or f.endswith('.jpg'):
                filename, _extension = os.path.splitext(os.path.basename(f))
                
                detect_traffic_signs(
                    name=filename,
                    image_path=path + f, 
                    outDir=outPath, 
                    save_all=verbose,
                    annot_dict=annot_dict
                )
        pass
    else:
        print(f"Path {path} is unrecognized. Please verify if path corresponds to valid file or directory.", file=sys.stderr)
