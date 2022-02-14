import cv2 as cv
import numpy as np
import argparse
import os
from datetime import datetime


class Autophotographer:
    def __init__(self, path_in, path_out, quantity, filename):
        self.path_in = path_in
        self.path_out = path_out
        self.num_to_output = quantity
        self.filename = filename

        self.find_best()

    def find_best(self):
        video = cv.VideoCapture(self.path_in)
        success, image = video.read()
        count = 0

        liked = []

        while success:
            if len(liked) < self.num_to_output:
                liked.append(count)

            # Try to read next frame and increase counter by 1
            success, image = video.read()
            count += 1

        print(f'Number of frames: {count}')

        self.save_best(video, liked)
        video.release()

    def save_best(self, video, img_list):
        for i in img_list:

            video.set(cv.CAP_PROP_POS_FRAMES, i)
            success, frame = video.read()

            if success:
                if not os.path.exists(self.path_out):
                    os.mkdir(self.path_out)

                path = rf'{self.path_out}{args.name}{i}.png'
                cv.imwrite(path, frame)
                print(f"Saved frame {i} to {path}")
            else:
                print("Unable to save frame!")
                break


# These algorithms take in an image and return a score based on the content of the image
class Algorithms:
    def __init__(self, image):
        self.image = image

    def average_brightness(self):
        pass

    def sharpness(self):
        pass


if __name__ == "__main__":
    args = argparse.ArgumentParser()

    args.add_argument("-i", "--pathIn", help="Path to Video File", required=True)
    args.add_argument("-o", "--pathOut", help="Path to output saved images to", default=rf'{os.getcwd()}/Output/')
    args.add_argument("-q", "--quantity", help="Number of images to output", default=5)
    args.add_argument("-n", "--name", help="The name of the files output (before file number is appended)",
                      default=datetime.now().strftime('%d-%m-%y_%H-%M-%S_'))

    args = args.parse_args()
    
    program = Autophotographer(args.pathIn, args.pathOut, args.quantity, args.name)
