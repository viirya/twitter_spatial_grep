 
import sys
import numpy
import timeit
import time
import os
import argparse

def parse_parameters():
 
    parser = argparse.ArgumentParser(description = 'Tools for hamming distance-based image retrieval by cuda')
    parser.add_argument('-f', help = 'Input filename.')
    parser.add_argument('-o', help = 'Output filename.')
    parser.add_argument('-batch_n', default = '1000', help = 'Number of lines in batch.')
    parser.add_argument('-bbox', help = 'Bounding box of geolocation.')
 
    args = parser.parse_args()

    return args


