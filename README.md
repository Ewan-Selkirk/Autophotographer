# Autophotographer
A program for turning a video feed into a user-defined number of images that fits the 'rule-of-thirds' rule of thumb.  \
Made with Python 3 & OpenCV. 

Created as a third year major project for Aberystwyth University by Ewan Selkirk.

## Requirements
- Python 3.5+ (*Developed with 3.8/3.10 but tested on 3.5-3.10*)
- pip 3
- Numpy
- OpenCV

## Installing & Running
1. Clone the repo with `git clone https://github.com/Ewan-Selkirk/Autophotographer.git` or download the repo as a zip
2. Install the requirements with `pip install -r requirements.txt`
3. Run the program with `python autophotographer.py -i [video file/folder path] -c [brightness, sharpness, colour/color]`

## Algorithms
### Brightness
The brightness algorithm attempts to find frames that match the rule-of-thirds by the brightness of the pixels inside
the 9 quadrants of an image.

### Sharpness
The sharpness algorithm attempts to find frames that match the rule-of-thirds by the number of edge pixels after a Canny
edge detection operation is run on each quadrant of an image.

### Color/Colour
The color/colour algorithm attempts to find frames that match the rule-of-thirds by using the 3 separate colour channels 
of each of the 9 quadrants of an image.

## Options
### Required
| Short-code | Long-code | Description                                                  | Choices                             |
|------------|-----------|--------------------------------------------------------------|-------------------------------------|
| -i         | --pathIn  | The file or directory of files to run the program on         | N/A                                 |
| -c         | --config  | A list of algorithms to run on the video (multiple accepted) | brightness, sharpness, colour/color |
|
### Optional
| Short-code | Long-code  | Description                                                                                                                                                               |
|------------|------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| -o         | --pathOut  | Set the folder to output the created images to. Defaults to an 'Output' folder in the current working directory.                                                          |
| -q         | --quantity | Sets the number of images to create.                                                                                                                                      |
| -n         | --name     | Sets the file name of every file created by the program. Allows for the use of tokens. Defaults to the date and time the operation was run at, followed by the file name. |
| -h         | --help     | Displays this help field.                                                                                                                                                 |
| -v         | --verbose  | Displays additional information while the program runs.                                                                                                                   |


### Name Tokens 
The name argument allows for the use of tokens to make cataloging images easier.  \
Below are the list of tokens that are currently supported: 

| Token | Example              | Description                                         |
|-------|----------------------|-----------------------------------------------------|
| %dt   | 04-22-10_15-04-22    | The current date and time                           |
| %d    | 04-22-10             | Only the current date                               |
| %t    | 15-04-22             | Only the current time                               |
| %f    | tram                 | The current file name (minus file extension)        |
| %o    | brightness-sharpness | The list of algorithms being ran on the files       |
| %os   | b-s                  | Shortened list of algorithms being ran on the files |
| %%    | %                    | An escaped percentage symbol                        |

## Developing
### Requirements
- Git
- Git LFS
- Python 3.5+
- pip 3
  - virtualenv (if you don't want packages installed globally)
  - opencv-contrib-python >= 4.5.5.60 or opencv-python >= 4.5.5.60
  - numpy >= 1.22

----------

