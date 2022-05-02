# traffic-signifier

## Requirements
To make sure you have all the requirements run: `pip install -r requirements.txt`
```
numpy>=1.22.3
opencv_python>=4.5.5.64
pandas>=1.4.2
tqdm>=4.64.0
```

## Usage
You can run `python traffic_signifier.py -h` for help on how to run the program.
```
usage: traffic_signifier.py [-h] [--out OUT] [--annotations ANNOTATIONS] [--verbose] [--stack] path

Detect traffic signs in the image specified. If input is a directory, detect traffic signs on all images in the
folder.

positional arguments:
  path                  Path of the image or directory to process.

options:
  -h, --help            show this help message and exit
  --out OUT, -o OUT     Path of output directory. Defaults to `out`
  --annotations ANNOTATIONS, -a ANNOTATIONS
                        Path to annotations directory for scoring accuracy of detection.
  --verbose, -v         Controls the verbosity, if enable the program will output more messages and save more images
                        related to the processing.
  --stack               Needs verbosity active. Stacks all images generated into the same image.
```
