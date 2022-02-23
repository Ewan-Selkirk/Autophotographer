import cv2 as cv
import numpy as np
import argparse
import os
from datetime import datetime

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


def as_percentage(value) -> float:
    if value < 0 or value > 255:
        raise ValueError("Value must be between 0-255")

    if value <= 127:
        return value / 127 * 100
    else:
        return 200 - (value / 127) * 100


def log(message):
    if verbose:
        print(f"[{datetime.now().strftime('%H:%M:%S')}]", message)


class Autophotographer:
    def __init__(self, args):
        self.path_in = args.pathIn
        self.path_out = args.pathOut
        self.num_to_output = args.quantity
        self.filename = args.name or str(f"{datetime.now().strftime('%d-%m-%y_%H-%M-%S_')}"
                                         f"{args.pathIn.split('/')[-1][:-4]}_")
        # self.config = args.config

        # Set the global verbose variable if the verbose argument is passed
        global verbose
        verbose = args.verbose

        self.find_best()

    def find_best(self):
        t_start = datetime.now()
        video = cv.VideoCapture(self.path_in)
        success, image = video.read()
        f_count = 0

        # Create an empty 2D array to store the frame number, and it's average brightness
        liked = [[]]

        while success:
            alg = Algorithms(image)
            avg = alg.average_brightness()

            # Initially fill the list with the first few frames
            if len(liked) < self.num_to_output:
                liked.append([f_count, avg])
                liked.sort(reverse=True)
            else:
                for c, v in liked:
                    if avg > v:
                        # Remove the last element of the array (the smallest value)
                        liked.pop()
                        liked.append([f_count, avg])
                        liked.sort(reverse=True)  # In theory this shouldn't be needed
                        break

            # Try to read next frame and increase counter by 1
            success, image = video.read()
            f_count += 1

        log(f'Number of frames: {f_count}')

        self.save_best(fetch_frames(video, liked))
        video.release()
        t_end = datetime.now()
        t_diff = (t_end - t_start).seconds
        log(f"Operation took {str(divmod(t_diff, 60)[0]) + ' minutes and ' if divmod(t_diff, 60)[0] > 0 else ''}"
            f"{divmod(t_diff, 60)[1]} seconds to complete")

    def save_best(self, images):
        for c, i in images:
            if not os.path.exists(self.path_out):
                os.mkdir(self.path_out)

            path = rf'{self.path_out}{self.filename}{c}.png'
            cv.imwrite(path, i)
        log(f"Saved {len(images)} images to {self.path_out}")

    def parse_config(self, config):
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
        return as_percentage(np.average(self.image))

    def sharpness(self):
        pass


if __name__ == "__main__":
    args = argparse.ArgumentParser()

    args.add_argument("-i", "--pathIn", help="Path to Video File", required=True)
    # args.add_argument("-c", "--config", help="", required=True,
    #                   choices=[], nargs="+")
    args.add_argument("-o", "--pathOut", help="Path to output saved images to", default=rf'{os.getcwd()}/Output/')
    args.add_argument("-q", "--quantity", help="Number of images to output", default=5)
    args.add_argument("-n", "--name", help="The name of the files output (before file number is appended)")
    args.add_argument("-v", "--verbose", help="Prints messages to the console window", action="store_true")

    program = Autophotographer(args.parse_args())
