import utils.image_processing as imp

import os

import cv2 as cv
import numpy as np

def preprocess_image(bgr_image):
    out_image = imp.hsv_clahe_equalization(
        bgr_image=bgr_image,
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    out_image = imp.automatic_brightness_contrast(
        bgr_image=out_image,
        clip_hist_percent=0.1,
        use_scale_abs=True
    )
    
    return out_image

def segment_reds(bgr_image):
    out_image = imp.meanShiftFiltering(bgr_image)
    hsv_image = cv.cvtColor(out_image, cv.COLOR_BGR2HSV)
    mask = imp.extract_red_hsv(hsv_image=hsv_image)
    out_image = cv.bitwise_and(out_image, out_image, mask=mask)
    out_image = imp.binarize(out_image, threshold_percent=0.75)
    out_image = imp.removeSmallComponents(out_image, size_threshold=200)
    return out_image

def segment_blues(bgr_image):
    out_image = imp.meanShiftFiltering(bgr_image)
    hsv_image = cv.cvtColor(out_image, cv.COLOR_BGR2HSV)
    mask = imp.extract_blue_hsv(hsv_image=hsv_image)
    out_image = cv.bitwise_and(out_image, out_image, mask=mask)
    out_image = imp.binarize(out_image, threshold_percent=0.75)
    out_image = imp.removeSmallComponents(out_image, size_threshold=200)
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

    if save_all:
        output = {
            f"{name}_processed": ("processed", processed_image),
            f"{name}_red_segment": ("red_segmentation", cv.cvtColor(red_image, cv.COLOR_GRAY2BGR)),
            f"{name}_blue_segment": ("blue_segmentation", cv.cvtColor(blue_image, cv.COLOR_GRAY2BGR)),
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
