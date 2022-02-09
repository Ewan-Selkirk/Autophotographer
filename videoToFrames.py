import cv2
import numpy as np
import argparse
import os


def find_best(path_in, path_out="Output/", num_to_output=5):
    frames = cv2.VideoCapture(path_in)
    success, image = frames.read()
    count = 0

    liked = {}

    while success:
        # Try to read next frame and increase counter by 1
        success, image = frames.read()
        count += 1

    # print({v for k, v in sorted(liked.items(), key=lambda item: item[1])})

    print(f'Number of frames: {count}')
    print(f'Saved {num_to_output} images [Not really]')

    frames.release()


if __name__ == "__main__":
    args = argparse.ArgumentParser()

    args.add_argument("-i", "--pathIn", help="Path to Video File", required=True)
    args.add_argument("-e", "--pathOut", help="Path to output saved images to", default="Output/")
    args.add_argument("-o", "--output", help="Number of images to output", default=5)

    args = args.parse_args()
    find_best(args.pathIn, args.pathOut, args.output)
