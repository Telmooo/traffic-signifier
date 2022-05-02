import utils.image_processing as imp
import utils.xml_parser as xp

import os
from collections import defaultdict

from tqdm import tqdm

import numpy as np
import cv2 as cv
import pandas as pd

def preprocess_image(bgr_image):
    out_image = imp.attenuate_sky_light(
        bgr_image=bgr_image,
        sky_threshold=0.40,
        max_blue_threshold=0.85
    )
    out_image = imp.hsv_clahe_equalization(
        bgr_image=out_image,
        clipLimit=3.0,
        tileGridSize=(15, 15)
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
    (brect_x, brect_y, brect_w, brect_h), contour, red_ratio, blue_ratio, white_black_ratio, red_blue_ratio, contourArea, sqp, trg, circle = roi

    contourPerimeter = cv.arcLength(contour, True)
    # Bounding Rectangle Area - used to calculate extent of contour
    brect_area = brect_w * brect_h
    extent = contourArea / (brect_area + 1e-6)

    # Minimum Bounding Rectangle - takes into account orientation and fits the bounding rectangle thighly
    (min_brect_x0, min_brect_y0), (min_brect_x1, min_brect_y1), min_brect_angle = cv.minAreaRect(contour)
    min_brect_area = abs(min_brect_x0 - min_brect_x1) * abs(min_brect_y0 - min_brect_y1)

    min_extent = contourArea / (min_brect_area + 1e-6)

    # Circularity - measures how compact the contour is
    circularity = (4 * np.pi * contourArea) / (contourPerimeter * contourPerimeter + 1e-6)

    # Minimum Enclosing Circle - similar to circularity
    (min_circle_x, min_circle_y), circle_radius = cv.minEnclosingCircle(contour)
    min_circle_area = np.pi * circle_radius * circle_radius

    circle_extent = contourArea / (min_circle_area + 1e-6)

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
            probability_table[i, j] = abs(ratio - class_ratio) # / (class_ratio + 1e-6)
        
        probability_table[i, n_classes] = np.sum(probability_table[i, 0:n_classes])
        probability_table[i, 0:n_classes] = (1 - (probability_table[i, 0:n_classes] / (probability_table[i, n_classes] + 1e-6))) #/ (n_classes - 1)
        probability_table[i, n_classes] = np.sum(probability_table[i, 0:n_classes])

    # Add corners row
    probability_table[n_metrics - 1, :n_classes] = np.array([circle, sqp, circle, trg])
    # probability_table[n_metrics - 1, n_classes] = np.sum(probability_table[n_metrics - 1, 0:n_classes])
    # probability_table[n_metrics - 1, :n_classes] /= (probability_table[n_metrics - 1, n_classes] + 1e-6)

    probability_table[-1, :n_classes] = np.mean(probability_table[:-1, :-1], axis=0)
    probability_table[-1, n_classes] = np.sum(probability_table[-1, 0:n_classes])

    CORNER_THRESHOLD = 0.60

    if roi_type == "red":
        max_corner = np.argmax(probability_table[metrics.index("corners"), :n_classes])
        corner_prob = probability_table[metrics.index("corners"), max_corner]
        if corner_prob < 0.40:
            circularity = np.mean([
                    probability_table[metrics.index("circularity"), 0],
                    probability_table[metrics.index("circle_extent"), 0]
                ])
            if circularity > 0.90:
                color_ratio = np.array([probability_table[metrics.index("color_ratio"), 0], probability_table[metrics.index("color_ratio"), 2]])
                max_color_ratio = np.argmax(color_ratio)
                max_color_ratio_prob = color_ratio[max_color_ratio]
                if max_color_ratio == 1 and max_color_ratio_prob > 0.8:
                    chosen_shape = "octagon"
                elif max_color_ratio == 0 and max_color_ratio_prob > 0.8:
                    chosen_shape = "circle"
                else:
                    chosen_shape = "other"
            else:
                chosen_shape = "other"
        else:
            if max_corner == 1: # quadrilateral
                chosen_shape = "other"
            elif (max_corner == 0 or max_corner == 2) and corner_prob > CORNER_THRESHOLD: # circle | octagon
                color_ratio = np.array([probability_table[metrics.index("color_ratio"), 0], probability_table[metrics.index("color_ratio"), 2]])
                max_color_ratio = np.argmax(color_ratio)
                max_color_ratio_prob = color_ratio[max_color_ratio]
                if max_color_ratio == 1 and max_color_ratio_prob > 0.8:
                    chosen_shape = "octagon"
                elif max_color_ratio == 0 and max_color_ratio_prob > 0.8:
                    chosen_shape = "circle"
                else:
                    circularity = np.mean(np.vstack([
                        probability_table[metrics.index("circularity"), :n_classes],
                        probability_table[metrics.index("circle_extent"), :n_classes]
                        ]), axis=0
                    )
                    max_circularity = np.argmax(circularity)
                    max_circularity_prob = circularity[max_circularity]

                    if max_circularity_prob > 0.8:
                        if max_circularity == 0:
                            chosen_shape = "circle"
                        elif max_circularity == 2:
                            chosen_shape = "octagon"
                        else:
                            chosen_shape = "other"
                    else:
                        chosen_shape = "other"
            elif corner_prob > CORNER_THRESHOLD: # triangle
                max_color_ratio = np.argmax(probability_table[metrics.index("color_ratio"), :n_classes])
                if max_color_ratio == 0 or max_color_ratio == 3: # circle | triangle
                    max_circle_extent = np.argmax(probability_table[metrics.index("circle_extent"), :n_classes])
                    if max_circle_extent == 0 or max_circle_extent == 2:
                        chosen_shape = "other"
                    else:
                        chosen_shape = "triangle"
                else:
                    chosen_shape = "other"
            else:
                chosen_shape = "other"

    elif roi_type == "blue":
        max_color_ratio = np.argmax(probability_table[metrics.index("color_ratio"), :n_classes])
        if max_color_ratio == 0 or max_color_ratio == 1: # circle | quadrilateral
            max_corner = np.argmax(probability_table[metrics.index("corners"), :n_classes])
            corner_prob = probability_table[metrics.index("corners"), max_corner]
            if corner_prob < 0.40:
                circularity = np.mean([
                        probability_table[metrics.index("circularity"), 0],
                        probability_table[metrics.index("circle_extent"), 0]
                    ])
                if circularity > 0.90:
                    chosen_shape = "circle"
                else:
                    chosen_shape = "other"
            else:
                if (max_corner == 0 or max_corner == 2) and corner_prob > CORNER_THRESHOLD: # circle | octagon
                    circularity = np.mean(np.vstack([
                        probability_table[metrics.index("circularity"), :n_classes],
                        probability_table[metrics.index("circle_extent"), :n_classes]
                        ]), axis=0
                    )
                    max_circularity = np.argmax(circularity)

                    if max_circularity == 0 or max_circularity == 2:
                        chosen_shape = "circle"
                    else:
                        chosen_shape = "other"

                elif max_corner == 1 and corner_prob > CORNER_THRESHOLD: # quadrilateral
                    avg_prob = np.mean(probability_table[:-1, :-1], axis=0)
                    quad_avg_prob = avg_prob[classes.index("quadrilateral")]
                    if quad_avg_prob > 0.7:
                        chosen_shape = "quadrilateral"
                    else:
                        chosen_shape = "other"
                elif corner_prob > CORNER_THRESHOLD: # triangle
                    avg_prob = np.mean(probability_table[:-1, :-1], axis=0)
                    max_avg_prob = np.argmax(avg_prob)
                    if max_avg_prob != 1:
                        chosen_shape = "other"
                    else: 
                        quad_avg_prob = avg_prob[classes.index("quadrilateral")]
                        if quad_avg_prob > 0.7:
                            chosen_shape = "quadrilateral"
                        else:
                            chosen_shape = "other"
                else:
                    chosen_shape = "other"
        else:
            chosen_shape = "other"

    if return_probabilities:
        df = pd.DataFrame(data=probability_table[:, :-1], columns=classes, index=metrics + ["AVG(P)"])
        return chosen_shape, df

    return chosen_shape

def save_images(outDir, output_dict):
    os.makedirs(outDir, exist_ok=True)

    for name, (subDir, image) in output_dict.items():
        directory = f"{outDir}/{subDir}"
        os.makedirs(directory, exist_ok=True)

        path = f"{directory}/{name}.png"
        cv.imwrite(path, image)

def cvt_annot_sign_type(type: str) -> str:
    return {
        'stop': 'stop',
        'speedlimit': 'prohibitory',
        'crosswalk': 'information',
        'trafficlight': '',
    }[type]


def detect_traffic_signs(name: str, image_path : str, outDir : str, annot_dict : dict, save_all : bool, stack_images : bool) -> list:
    signs_identified = []
    if (annot_dict):
        signs_identified = [
            {
                'image': name,
                'index': i,
                'class': cvt_annot_sign_type(a['name']),
                'identified': False,
                'xmin': a['xmin'],
                'ymin': a['ymin'],
                'w': a['xmax'] - a['xmin'],
                'h': a['ymax'] - a['ymin'],
            } for (i, a) in enumerate(annot_dict[name]) if a['name'] != 'trafficlight']
        if (len(signs_identified) == 0):
            return []

    image = cv.imread(image_path)
    
    processed_image = preprocess_image(image)

    # Old Segmentation - prone to intense blues on the sky
    # red_image = imp.segment_reds(processed_image)
    # blue_image = imp.segment_blues(processed_image)

    red_image, blue_image, white_black_image = imp.segment(processed_image)

    red_edges = imp.edge_detection(red_image)
    blue_edges = imp.edge_detection(blue_image)

    red_roi = imp.extractROI(red_edges, red_image=red_image, blue_image=blue_image, white_black_image=white_black_image, roi_type="red")
    blue_roi = imp.extractROI(blue_edges, red_image=red_image, blue_image=blue_image, white_black_image=white_black_image, roi_type="blue")

    roi_red_image = np.zeros(shape=(red_edges.shape + (3,)), dtype=np.uint8)
    for roi in red_roi:
        (x, y, w, h), contours, _, _, _, _, _ = roi
        cv.rectangle(roi_red_image, (int(x), int(y)), (int(x+w), int(y+h)), (197, 183, 255), 1)
        cv.drawContours(roi_red_image, [contours], 0, color=(255, 255, 255))

    roi_blue_image = np.zeros(shape=(blue_edges.shape + (3,)), dtype=np.uint8)
    for roi in blue_roi:
        (x, y, w, h), contours, _, _, _, _, _ = roi
        cv.rectangle(roi_blue_image, (int(x), int(y)), (int(x+w), int(y+h)), (197, 183, 255), 1)
        cv.drawContours(roi_blue_image, [contours], 0, color=(255, 255, 255))

    # Get corner info
    for i in range(len(red_roi)):
        roi = red_roi[i]
        (brect_x, brect_y, brect_w, brect_h), contour, red_ratio, blue_ratio, white_black_ratio, red_blue_ratio, contourArea = roi
        sqp, trg, circle = imp.corner_detection(roi)
        red_roi[i] = ((brect_x, brect_y, brect_w, brect_h), contour,  red_ratio, blue_ratio, white_black_ratio, red_blue_ratio, contourArea, sqp, trg, circle)

    for i in range(len(blue_roi)):
        roi = blue_roi[i]
        (brect_x, brect_y, brect_w, brect_h), contour, red_ratio, blue_ratio, white_black_ratio, red_blue_ratio, contourArea = roi
        sqp, trg, circle = imp.corner_detection(roi)
        blue_roi[i] = ((brect_x, brect_y, brect_w, brect_h), contour, red_ratio, blue_ratio, white_black_ratio, red_blue_ratio, contourArea, sqp, trg, circle)

    output_image = image.copy()

    def check_result(signs: list[dict], o_class: str, x: int, y: int, w: int, h: int):
        ROI_MARGIN = 20
        for s in signs:
            if s['class'] == o_class and (
                    s['xmin'] - ROI_MARGIN < x and
                    s['ymin'] - ROI_MARGIN < y and
                    s['w'] + 2 * ROI_MARGIN > w and
                    s['h'] + 2 * ROI_MARGIN > h
                ):
                s['identified'] = True
                return

    for roi in red_roi:
        shape = detect_shape(roi, "red", return_probabilities=False)
        (x, y, w, h), contours, red_ratio, blue_ratio, white_black_ratio, red_blue_ratio, contourArea, sqp, trg, circle = roi

        if shape != "other":
            cv.drawContours(output_image, [contours], 0, color=(25, 255, 40), thickness=2)
            output_class = output_classes[(shape, "red")]
            cv.putText(output_image, output_class, (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.6, (25, 255, 40), 2, cv.LINE_AA)
            check_result(signs_identified, output_class, x, y, w, h)

    for roi in blue_roi:
        shape = detect_shape(roi, "blue", return_probabilities=False)
        (x, y, w, h), contours, red_ratio, blue_ratio, white_black_ratio, red_blue_ratio, contourArea, sqp, trg, circle = roi
        if shape != "other":
            cv.drawContours(output_image, [contours], 0, color=(25, 255, 40), thickness=2)
            output_class = output_classes[(shape, "blue")]
            cv.putText(output_image, output_class, (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.6, (25, 255, 40), 2, cv.LINE_AA)
            check_result(signs_identified, output_class, x, y, w, h)

    if save_all:
        if stack_images:
            hstack_1 = cv.hconcat([processed_image, cv.cvtColor(red_image, cv.COLOR_GRAY2BGR), cv.cvtColor(red_edges, cv.COLOR_GRAY2BGR), roi_red_image])
            hstack_2 = cv.hconcat([cv.cvtColor(blue_image, cv.COLOR_GRAY2BGR), cv.cvtColor(blue_edges, cv.COLOR_GRAY2BGR), roi_blue_image, output_image])
            final_stack = cv.vconcat([hstack_1, hstack_2])
            output = {
                f"{name}_stack": ("stack", final_stack),
                f"{name}_output": ("output", output_image),
            }
        else:
            output = {
                f"{name}_processed": ("processed", processed_image),
                f"{name}_red_segment": ("red_segmentation", cv.cvtColor(red_image, cv.COLOR_GRAY2BGR)),
                f"{name}_blue_segment": ("blue_segmentation", cv.cvtColor(blue_image, cv.COLOR_GRAY2BGR)),
                f"{name}_red_edges": ("red_edges", cv.cvtColor(red_edges, cv.COLOR_GRAY2BGR)),
                f"{name}_blue_edges": ("blue_edges", cv.cvtColor(blue_edges, cv.COLOR_GRAY2BGR)),
                f"{name}_red_roi": ("red_roi", roi_red_image),
                f"{name}_blue_roi": ("blue_roi", roi_blue_image),
                f"{name}_output": ("output", output_image),
            }
    else:
        output = {
            f"{name}_output": ("output", output_image),
        }

    save_images(outDir, output)

    return signs_identified

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
        "--out", "-o", help="Path of output directory. Defaults to `out`", default="./out"
    )

    parser.add_argument(
        "--annotations", "-a", help="Path to annotations directory for scoring accuracy of detection."
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Controls the verbosity, if enable the program will output more messages and save more images related to the processing."
    )

    parser.add_argument(
        "--stack", action="store_true",
        help="Needs verbosity active. Stacks all images generated into the same image."
    )

    args = parser.parse_args()

    path = args.path
    outPath = args.out
    annotationsPath = args.annotations
    if (annotationsPath[-1] != os.path.sep):
        annotationsPath += os.path.sep
    
    verbose = args.verbose
    stackImages = args.stack
    isFile = os.path.isfile(path)
    isDir = os.path.isdir(path)
    
    annot_dict = {}
    if isFile:
        filename, _extension = os.path.splitext(os.path.basename(path))
        
        if annotationsPath:
            annot_dict = xp.parse(f'{annotationsPath}{filename}.xml')
            
        results = detect_traffic_signs(
            name=filename,
            image_path=path,
            outDir=outPath,
            save_all=verbose,
            stack_images=stackImages,
            annot_dict=annot_dict
        )
        os.makedirs(os.path.join(outPath, 'results'), exist_ok=True)
        pd.DataFrame([results]).set_index(['image', 'index']).to_csv(os.path.join(outPath, 'results', f'{filename}.csv'))

    elif isDir:
        
        if annotationsPath:
            annot_dict = xp.from_dir(annotationsPath)
        
        dir_files = os.listdir(path)
        results = []
        for file in tqdm(os.listdir(path)):
            if file.endswith('.png') or file.endswith('.jpg'):
                filename, _extension = os.path.splitext(os.path.basename(file))
                
                results += detect_traffic_signs(
                    name=filename,
                    image_path=os.path.join(path, file),
                    outDir=outPath, 
                    save_all=verbose,
                    stack_images=stackImages,
                    annot_dict=annot_dict
                )
        
        pd.DataFrame(results).set_index(['image', 'index']).to_csv(os.path.join(outPath, 'results.csv'))
    else:
        print(f"Path {path} is unrecognized. Please verify if path corresponds to valid file or directory.", file=sys.stderr)
