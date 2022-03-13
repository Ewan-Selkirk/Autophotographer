import cv2 as cv
import numpy as np
import argparse
import os
import re
import enum

from datetime import datetime

verbose = False
owner = None


# An Enum for easily accessing the row and columns we need from a 3x3 grid
# (Use enum.Enum to avoid an `Enum` variable from being created)
class Array_Vector(enum.Enum):
    TOP = (0, 1, 2)
    LEFT = (0, 3, 6)
    BOTTOM = (6, 7, 8)
    RIGHT = (2, 5, 8)
    # "You can't half a third!" blah blah blah fight me
    TOP_HALF = (0, 1, 2, 3, 4, 5)
    BOTTOM_HALF = (3, 4, 5, 6, 7, 8)
    LEFT_HALF = (0, 1, 3, 4, 6, 7)
    RIGHT_HALF = (1, 2, 4, 5, 7, 8)


# The current implementation takes the 2D array of 'good' images and
# replaces the second element (originally the average score in this instance)
# with the image array
# TODO: Find a better way to do this
def fetch_frames(video, img_list) -> []:
    for v in range(len(img_list)):
        video.set(cv.CAP_PROP_POS_FRAMES, img_list[v][0])
        success, frame = video.read()

        if success:
            img_list[v][1] = frame

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
        return round(200 - (value / (r_max // 2)) * 100)


# Print messages to the console only if the '-v' argument is passed at the start
# Prepends a timestamp to every message
# Optional `parent` variable will print where the message is coming from (E.G. Which file is currently
# being operated on)
# (The owner variable will retain its value until it is changed or reset to nothing)
def log(message, parent=None) -> None:
    if verbose:
        global owner
        if parent is not None and parent != owner:
            owner = parent
        print(f"[{datetime.now().strftime('%H:%M:%S')}]", f"{owner + ': ' if owner is not None else ''}{message}")


# Return a private function which returns 3 values from the provided array
def get_array_vector(array) -> []:
    return lambda t: [array[t[i]] for i in range(len(t))] if type(t) is (tuple or list) else ValueError("")


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
        if re.search(r"\.\w{3}", self.path_in):
            try:
                filename = self.path_in.split('/')[-1]
                self.video.append([filename, cv.VideoCapture(self.path_in)])
                log(f"Using file: {filename} to operate on!")
            except cv.error as err:
                raise Exception(err)
        else:
            # Create an ImageCapture object for any image files found in the directory
            images = ImageCapture(None)

            for filename in os.listdir(self.path_in):
                if filename.endswith((".png", ".jpg")):

                    if images.get(cv.CAP_PROP_FRAME_COUNT) != len([x for x in os.listdir(self.path_in)
                                                                   if re.search(r"\.\w{3}", x)]):
                        images.open([os.path.join(self.path_in, x) for x in os.listdir(self.path_in)
                                     if re.search(r"\.\w{3}", x)])
                        self.video.append([self.path_in.split("/")[-2], images])
                else:
                    self.video.append([filename, cv.VideoCapture(os.path.join(self.path_in, filename))])

            log(f"Found {len(self.video)} {'files' if len(self.video) > 1 else 'file'} to operate on!", parent=None)

        self.find_best()

    def find_best(self):
        t_start = datetime.now()

        for filename, video in self.video:
            success, image = video.read()
            f_count = 0

            if not success:
                raise FileNotFoundError("Unable to read frames from input video file!")

            # For every frame in the video, create an element for storing the scores
            # used to rate images
            score = {f: {"blurry": None, "thirdsy": None, c: None}
                     for f in range(int(video.get(cv.CAP_PROP_FRAME_COUNT)))
                     for c in self.config}

            while success:
                log(f"Working on frame {f_count}", parent=filename)
                img_rot = Rule_of_Thirds(image)

                score[f_count]["blurry"] = np.sum([img_rot.run_algorithm("is_blurry")]) / 9 * 100
                score[f_count]["thirdsy"] = np.std([[*img_rot.calc_diff(self.config)[c].values()]
                                                   for c in self.config])

                for c in self.config:
                    score[f_count][c] = np.mean(img_rot.run_algorithm(c))

                # Try to read next frame and increase counter by 1
                success, image = video.read()
                f_count += 1

            data = []
            for k, v in score.items():
                data.append([k, np.std([*v.values()])])

            data.sort(key=lambda x: x[1], reverse=True)
            self.save_best(fetch_frames(video, data[:self.num_to_output]))

            video.release()

        t_end = datetime.now()
        t_diff = (t_end - t_start).seconds
        log(f"Operation took {str(divmod(t_diff, 60)[0]) + 'minutes and ' if divmod(t_diff, 60)[0] > 0 else ''}"
            f"{divmod(t_diff, 60)[1]} seconds to complete")

    def save_best(self, images):
        for c, i in images:
            if not os.path.exists(self.path_out):
                os.mkdir(self.path_out)

            path = os.path.join(self.path_out, self.filename + str(c) + '.png')
            cv.imwrite(path, i)
        log(f"Saved {len(images)} images to {self.path_out}")

    def parse_config(self, algorithm_obj):
        for c in self.config:
            # Use reflection to get pointers to the algorithm functions*-
            return algorithm_obj.__getattribute__(c)


# These algorithms take in an image and return a score based on the content of the image
class Algorithms:
    def __init__(self, image):
        self.image = image

    def brightness(self) -> float:
        result = np.average(self.image)
        return result

    def color(self) -> float:
        pass

    def colour(self) -> float:
        return self.color()

    def sharpness(self):
        pass

    def is_blurry(self, threshold=200) -> bool:
        return cv.Laplacian(self.image, cv.CV_64F).var() < threshold


class Rule_of_Thirds:
    def __init__(self, split_image):
        self.image = split_image
        self.__split_img = []

        self.__split()

    def __split(self):
        # log("Splitting image into 9 chunks...")

        shape = self.image.shape

        # Method for splitting images into chunks based on this StackOverflow answer
        # https://stackoverflow.com/a/47581978
        # TODO: Fix issues with images that don't equally divide by 3
        self.__split_img = [
            self.image[x: x + shape[0] // 3, y: y + shape[1] // 3]
            for x in range(0, shape[0], shape[0] // 3)
            for y in range(0, shape[1], shape[1] // 3)
        ]

    def get_image_splits(self) -> []:
        return self.__split_img

    def run_algorithm(self, algorithm) -> []:
        # Use reflection to call the method on a brand new `Algorithms` instance
        return [Algorithms(s).__getattribute__(algorithm)() for s in self.__split_img]

    def setup_calc(self, config):
        data = {c: {} for c in config}
        vector = {c: get_array_vector(self.run_algorithm(c)) for c in config}

        return data, vector

    # def calc_mean(self, config):
    #     data, vector = self.setup_calc(config)
    #
    #     for c in vector:
    #         for e in Array_Vector:
    #             data[c][e.name] = np.mean([vector[c](e.value)])
    #
    #     return data

    def calc_diff(self, config: list[str]):
        data, _ = self.setup_calc(config)

        for c in config:
            tmp = self.run_algorithm(c)

            for v in Array_Vector:
                data[c][v.name] = []

                data[c][v.name].append(abs(tmp[v.value[0]] - tmp[v.value[1]]))
                data[c][v.name].append(abs(tmp[v.value[0]] - tmp[v.value[2]]))
                data[c][v.name].append(abs(tmp[v.value[1]] - tmp[v.value[2]]))

                data[c][v.name] = np.sum(data[c][v.name])

        return data


class ImageCapture:
    def __init__(self, filename):
        self.stream = []
        self.position = 0

        self.open(filename)

    def read(self) -> tuple[bool, np.ndarray or None]:
        success = False
        data = None

        if self.position < len(self.stream):
            success = True
            data = self.stream[self.position]
            self.position += 1

        return success, data

    def get(self, prop):
        if prop is cv.CAP_PROP_FRAME_COUNT:
            return len(self.stream)
        elif prop is cv.CAP_PROP_POS_FRAMES:
            return self.position
        else:
            pass

    def open(self, filename):
        if type(filename) is list:
            for img in filename:
                self.stream.append(cv.imread(img))
        elif type(filename) is str:
            if str(filename).endswith((".png", ".jpg")):
                self.stream.append(cv.imread(filename))
            else:
                for f in os.listdir(filename):
                    self.stream.append(cv.imread(os.path.join(filename, f)))

    def release(self):
        self.stream = []
        self.position = 0


if __name__ == "__main__":
    args = argparse.ArgumentParser()

    args.add_argument("-i", "--pathIn", help="Path to either a directory of video files, or a specific video file.",
                      required=True)
    args.add_argument("-c", "--config", help="List of operations to perform on the input video file(s)", required=True,
                      choices=["brightness", "sharpness", "color", "colour"], nargs="+")
    args.add_argument("-o", "--pathOut", help="Path to output saved images to (will default to an 'Output' folder in "
                                              "the current working directory)", default=rf'{os.getcwd()}/Output/')
    args.add_argument("-q", "--quantity", help="Number of images to output", default=5)
    args.add_argument("-n", "--name", help="The name of the files output (will default to the date and time, and the "
                                           "original filename if left out)")
    args.add_argument("-v", "--verbose", help="Prints messages to the console window", action="store_true")

    program = Autophotographer(args.parse_args())
