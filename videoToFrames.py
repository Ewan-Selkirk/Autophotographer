import cv2 as cv
import numpy as np
import argparse
import os
import re
from datetime import datetime
import math

verbose = False


# The current implementation takes the 2D array of 'good' images and
# replaces the second element (originally the average score in this instance)
# with the image array
# TODO: Find a better way to do this
def fetch_frames(video, img_list) -> []:
    for i in img_list:
        video.set(cv.CAP_PROP_POS_FRAMES, i[0])
        success, frame = video.read()

        if success:
            i[1] = frame

    return img_list


def export_data(name, data):
    f = open(name, "w")
    for d in data:
        f.write(f"{d}\n")

    f.close()


# Takes an input value and maps it to a user defined range,
# return the percentage
def as_percentage(value, r_min, r_max) -> float:
    if value < r_min or value > r_max:
        raise ValueError(f"Value must be between {r_min}-{r_max}")

    if value <= r_max // 2:
        return value / (r_max // 2) * 100
    else:
        return math.ceil(200 - (value / (r_max // 2)) * 100)


# Print messages to the console only if the '-v' argument is passed at the start
# Prepends a timestamp to every message
# Optional owner variable will print where the message is coming from (E.G. Which file is currently
# being operated on)
def log(message, owner='') -> None:
    if verbose:
        print(f"[{datetime.now().strftime('%H:%M:%S')}]", f"{owner + ': ' if owner != '' else ''}{message}")


# Return a private function which returns 3 values from the provided array
def get_array_vector(array) -> []:
    return lambda a, b, c: [array[a], array[b], array[c]]


class Autophotographer:
    def __init__(self, p_args):
        # Create an array to store our loaded video files in
        self.video = []

        # Set our console arguments as variables
        self.path_in = p_args.pathIn
        self.path_out = p_args.pathOut
        self.num_to_output = p_args.quantity
        self.filename = p_args.name or str(f"{datetime.now().strftime('%d-%m-%y_%H-%M-%S_')}"
                                           f"{p_args.pathIn.split('/')[-1][:-4]}_")
        self.config = p_args.config

        # Set the global verbose variable if the verbose argument is passed
        global verbose
        verbose = p_args.verbose

        self.start()

    def start(self):
        if not os.path.exists(self.path_in):
            raise FileNotFoundError("")

        # Check if the input path ends with a file extension using Regular Expression
        if re.search(r"\.[a-z]{3}", self.path_in):
            try:
                f = self.path_in.split('/')[-1]
                self.video.append([f, cv.VideoCapture(self.path_in)])
                log(f"Using file: {f} to operate on!")
            except cv.error as err:
                raise Exception(err)
        else:
            for filename in os.listdir(self.path_in):
                self.video.append([filename, cv.VideoCapture(os.path.join(self.path_in, filename))])

            log(f"Found {len(self.video)} {'files' if len(self.video) > 1 else 'file'} to operate on!")

        self.find_best()

    def find_best(self):
        t_start = datetime.now()

        for filename, video in self.video:
            success, image = video.read()
            f_count = 0

            # Create an empty 2D array to store the frame number, and it's average brightness
            liked = []

            if not success:
                raise FileNotFoundError("Unable to read frames from input video file!")

            while success:
                # Create an instance of the Algorithms class with the current frame as the input
                alg = Algorithms(image)

                # If the video frame is blurry:
                if alg.is_blurry(100):
                    # Skip to the next frame
                    success, image = video.read()
                    break
                else:
                    s_img = alg.rot_split()
                    data = [self.parse_config(alg)]

                    # Initially fill the list with the first few frames
                    if len(liked) < self.num_to_output:
                        liked.append([f_count, np.average(data)])
                        liked.sort(key=lambda x: x[1], reverse=True)
                    else:
                        for c, v in liked:
                            if np.average(data) > v:
                                # Remove the last element of the array (the smallest value)
                                liked.pop()
                                liked.append([f_count, np.average(data)])
                                # Reverse sort the array by the value of the second
                                liked.sort(key=lambda x: x[1], reverse=True)
                                break

                # Try to read next frame and increase counter by 1
                success, image = video.read()
                f_count += 1

            log(f'Number of frames: {f_count}', owner=filename)

            self.save_best(fetch_frames(video, liked), filename)
            video.release()

        t_end = datetime.now()
        t_diff = (t_end - t_start).seconds
        log(f"Operation took {str(divmod(t_diff, 60)[0]) + 'minutes and ' if divmod(t_diff, 60)[0] > 0 else ''}"
            f"{divmod(t_diff, 60)[1]} seconds to complete")

    def save_best(self, images, file):
        for c, i in images:
            if not os.path.exists(self.path_out):
                os.mkdir(self.path_out)

            path = rf'{self.path_out}{self.filename}{c}.png'
            cv.imwrite(path, i)
        log(f"Saved {len(images)} images to {self.path_out}", owner=file)

    def parse_config(self, algorithm_obj):
        for c in self.config:
            if c == "brightness":
                return algorithm_obj.average_brightness()
            elif c == "sharpness":
                pass

    # These algorithms take in an image and return a score based on the content of the image


class Algorithms:
    def __init__(self, image):
        self.image = image

    def rot_split(self):
        shape = self.image.shape

        # Method for splitting images into chunks based on this StackOverflow answer
        # https://stackoverflow.com/a/47581978
        return [
            self.image[x: x + shape[0] // 3, y: y + shape[1] // 3]
            for x in range(0, shape[0], shape[0] // 3)
            for y in range(0, shape[1], shape[1] // 3)
        ]

    def average_brightness(self) -> float:
        return as_percentage(np.average(self.image), 0, 255)

    def sharpness(self):
        pass

    def is_blurry(self, threshold) -> bool:
        return cv.Laplacian(self.image, cv.CV_64F).var() < threshold


if __name__ == "__main__":
    args = argparse.ArgumentParser()

    args.add_argument("-i", "--pathIn", help="Path to either a directory of video files, or a specific video file.",
                      required=True)
    args.add_argument("-c", "--config", help="List of operations to perform on the input video file(s)", required=True,
                      choices=["brightness", "sharpness"], nargs="+")
    args.add_argument("-o", "--pathOut", help="Path to output saved images to (will default to an 'Output' folder in "
                                              "the current working directory)", default=rf'{os.getcwd()}/Output/')
    args.add_argument("-q", "--quantity", help="Number of images to output", default=5)
    args.add_argument("-n", "--name", help="The name of the files output (will default to the date and time, and the "
                                           "original filename if left out)")
    args.add_argument("-v", "--verbose", help="Prints messages to the console window", action="store_true")

    program = Autophotographer(args.parse_args())
