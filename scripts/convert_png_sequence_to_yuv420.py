#!/usr/bin/env python3

import argparse

from yuvtools import yuvtools


def convert_png_sequence_to_yuv420():
    parser = argparse.ArgumentParser()
    parser.add_argument('file_pattern', help="pattern for png file search, e.g. 'png/*.png'", type=str)
    parser.add_argument('out_filename', help='path to output yuv 420 file', type=str)

    args = parser.parse_args()

    yuvtools.convert_png_sequence_to_yuv420(args.file_pattern, args.out_filename)
