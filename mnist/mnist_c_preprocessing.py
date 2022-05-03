# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 10:42:55 2022

@author: JeanMichelAmath
"""

import os
from pathlib import Path

MNIST_C = Path(os.getcwd(), '../data/mnist_c')
MNIST_C_LEFTOVERS = Path(os.getcwd(), '../data/mnist_c_leftovers')

identity_dir = os.path.join(MNIST_C, 'identity')
brightness_dir = os.path.join(MNIST_C, 'brightness')
canny_edges_dir = os.path.join(MNIST_C, 'canny_edges')
dotted_line_dir = os.path.join(MNIST_C, 'dotted_line')
fog_dir = os.path.join(MNIST_C, 'fog')
glass_blur_dir = os.path.join(MNIST_C, 'glass_blur')
impulse_noise_dir = os.path.join(MNIST_C, 'impulse_noise')
motion_blur_dir = os.path.join(MNIST_C, 'motion_blur')
rotate_dir = os.path.join(MNIST_C, 'rotate')
scale_dir = os.path.join(MNIST_C, 'scale')
shear_dir = os.path.join(MNIST_C, 'shear')
shot_noise_dir = os.path.join(MNIST_C, 'shot_noise')
spatter_dir = os.path.join(MNIST_C, 'spatter')
stripe_dir = os.path.join(MNIST_C, 'stripe')
translate_dir = os.path.join(MNIST_C, 'translate')
zigzag_dir = os.path.join(MNIST_C, 'zigzag')

contrast_dir = os.path.join(MNIST_C_LEFTOVERS, 'contrast')
defocus_blur_dir = os.path.join(MNIST_C_LEFTOVERS, 'defocus_blur')
elastic_transform_dir = os.path.join(MNIST_C_LEFTOVERS, 'elastic_transform')
frost_dir = os.path.join(MNIST_C_LEFTOVERS, 'frost')
gaussian_blur_dir = os.path.join(MNIST_C_LEFTOVERS, 'gaussian_blur')
gaussian_noise_dir = os.path.join(MNIST_C_LEFTOVERS, 'gaussian_noise')
inverse_dir = os.path.join(MNIST_C_LEFTOVERS, 'inverse')
jpeg_compression_dir = os.path.join(MNIST_C_LEFTOVERS, 'jpeg_compression')
line_dir = os.path.join(MNIST_C_LEFTOVERS, 'line')
pessimal_noise_dir = os.path.join(MNIST_C_LEFTOVERS, 'pessimal_noise')
pixelate_dir = os.path.join(MNIST_C_LEFTOVERS, 'pixelate')
quantize_dir = os.path.join(MNIST_C_LEFTOVERS, 'quantize')
saturate_dir = os.path.join(MNIST_C_LEFTOVERS, 'saturate')
snow_dir = os.path.join(MNIST_C_LEFTOVERS, 'snow')
speckle_noise_dir = os.path.join(MNIST_C_LEFTOVERS, 'speckle_noise')
zoom_blur_dir = os.path.join(MNIST_C_LEFTOVERS, 'zoom_blur')


corruptions = ['identity', 'brightness', 'canny_edges',
    'dotted_line', 'fog', 'glass_blur', 'impulse_noise', 
    'motion_blur', 'rotate', 'scale', 'shear', 'shot_noise',
    'spatter', 'stripe', 'translate', 'zigzag', 'contrast',
    'defocus_blur', 'elastic_transform', 'frost', 'gaussian_blur',
    'gaussian_noise', 'inverse', 'jpeg_compression', 'line',
    'pessimal_noise', 'pixelate', 'quantize', 'saturate',
    'snow', 'speckle_noise', 'zoom_blur']

corruption_dir = [identity_dir, brightness_dir, canny_edges_dir, dotted_line_dir,
                  fog_dir, glass_blur_dir, impulse_noise_dir, motion_blur_dir,
                  rotate_dir, scale_dir, shear_dir, shot_noise_dir, spatter_dir,
                  stripe_dir, translate_dir, zigzag_dir, contrast_dir, defocus_blur_dir,
                  elastic_transform_dir, frost_dir, gaussian_blur_dir, gaussian_noise_dir,
                  inverse_dir, jpeg_compression_dir, line_dir, pessimal_noise_dir, pixelate_dir,
                  quantize_dir, saturate_dir, snow_dir, speckle_noise_dir, zoom_blur_dir]

