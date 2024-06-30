import numpy as np
from scipy.ndimage.filters import convolve
import cv2
from gaussian_filter import gaussian_filter

def generate_octave(init_level, s, sigma):
	octave = [init_level]

	k = 2**(1/s)
	kernel = gaussian_filter(k * sigma)

	for i in range(s+2):
		next_level = convolve(octave[-1], kernel)
		next_level = cv2.resize(next_level, octave[0].shape[::-1])
		octave.append(next_level)

	return octave

def generate_gaussian_pyramid(im, num_octave, s, sigma):
    pyr = []

    for _ in range(num_octave):
        octave = generate_octave(im, s, sigma)
        pyr.append(octave)
        im = octave[-3][::2, ::2]

    return pyr