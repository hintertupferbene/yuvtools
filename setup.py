#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='yuvtools',
      version='1.0',
      description='Tools for loading, converting, and evaluating yuv video fules',
      author='Andreas Heindel',
      author_email='andreas.heindel@fau.de',
      url='https://github.com/hintertupferbene/yuvtools',
      packages=find_packages(),
      entry_points={
          'console_scripts': [
              'convert_png_sequence_to_yuv420 = scripts.convert_png_sequence_to_yuv420:convert_png_sequence_to_yuv420'
          ]}
      )
