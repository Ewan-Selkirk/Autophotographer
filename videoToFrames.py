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
    if not value < 0 and not value > 255:
        if value < (256 / 2) - 1:
            return value / ((256 / 2) - 1) * 100
        else:
            return (255 - value) / ((256 / 2) - 1) * 100
    else:
        raise ValueError("Value must be between 0-255")


class Autophotographer:
    def __init__(self, args):
        self.path_in = args.pathIn
        self.path_out = args.pathOut
        self.num_to_output = args.quantity
        self.name = args.name
        # self.config = args.config

        self.find_best()

    def find_best(self):
        video = cv.VideoCapture(self.path_in)
        success, image = video.read()
        f_count = 0

        # Create an empty 2D array to store the frame number and it's average brightness
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

        print(f'Number of frames: {f_count}')

        self.save_best(fetch_frames(video, liked))
        video.release()

    def save_best(self, images):
        for c, i in images:
            if not os.path.exists(self.path_out):
                os.mkdir(self.path_out)

            path = rf'{self.path_out}{self.name}{c}.png'
            cv.imwrite(path, i)
            # print(f"Saved frame {i} to {path}")

    def parse_config(self, config):
        pass

    # These algorithms take in an image and return a score based on the content of the image


class Algorithms:
    def __init__(self, image):
        self.image = image

    def average_brightness(self) -> float:
        return np.average(self.image)

    def sharpness(self):
        pass


if __name__ == "__main__":
    args = argparse.ArgumentParser()

    args.add_argument("-i", "--pathIn", help="Path to Video File", required=True)
    # args.add_argument("-c", "--config", help="", required=True,
    #                   choices=[], nargs="+")
    args.add_argument("-o", "--pathOut", help="Path to output saved images to", default=rf'{os.getcwd()}/Output/')
    args.add_argument("-q", "--quantity", help="Number of images to output", default=5)
    args.add_argument("-n", "--name", help="The name of the files output (before file number is appended)",
                      default=datetime.now().strftime('%d-%m-%y_%H-%M-%S_'))
    args.add_argument("-v", "--verbose", help="Prints messages to the console window", action="store_true")

    program = Autophotographer(args.parse_args())
