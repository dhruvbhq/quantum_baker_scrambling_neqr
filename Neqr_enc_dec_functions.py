#!/usr/bin/env python
# coding: utf-8

import numpy as np
#from PIL import Image
#from PIL import ImageOps
import random

# Some helper functions
def get_normalized_state(state_vector):
    norm = np.linalg.norm(state_vector)
    if(norm != 0):
        state_vector = state_vector / norm
    else:
        print("ERROR, get_normalized_state")
    return state_vector

def get_pixel_state_vector(pixel_val, pixel_depth):
    if(pixel_val > 2 ** pixel_depth):
        print("ERROR, get_pixel_state_vector")
    else:
        #state_vector = np.zeros((2 ** pixel_depth)) + np.zeros((2 ** pixel_depth))*1.0j
        state_vector = np.zeros((2 ** pixel_depth))
        state_vector[pixel_val] = 1
        state_vector = get_normalized_state(state_vector)
        return state_vector
    
def get_position_state_vector(x, y, x_dim, y_dim):
    # x_dim and y_dim are expected to be in powers of 2
    if((x > x_dim - 1) or (y > y_dim - 1)):
        print("ERROR, get_position_state_vector")
    else:
        #state_vector = np.zeros((x_dim * y_dim)) + np.zeros((x_dim * y_dim))*1.0j
        state_vector = np.zeros((x_dim * y_dim))
        state_vector[y_dim*x + y] = 1
        state_vector = get_normalized_state(state_vector)
        return state_vector

def simple_neqr_encoder(img_input, pixel_depth):
    # Accepts the input image and converts it to a numpy array
    img_np     = np.array(img_input)
    img_dim    = np.ndim(img_np)
    img_shape  = np.shape(img_np)
    img_y_max  = img_shape[0]
    img_x_max  = img_shape[1]
    enc_dim    = img_x_max * img_y_max * (2 ** pixel_depth)
    #enc_output = np.zeros((enc_dim)) + np.zeros((enc_dim))*1.0j 
    enc_output = np.zeros((enc_dim))
    for x_idx in range(img_x_max):
        for y_idx in range(img_y_max):
            pixel_val = img_np[y_idx, x_idx]
            enc_output = enc_output + np.kron(get_pixel_state_vector(pixel_val, pixel_depth),
                                              get_position_state_vector(x_idx, y_idx, img_x_max, img_y_max))
            
    enc_output = get_normalized_state(enc_output)
    return(enc_output)

def neqr_decoder(enc_input, pixel_depth, x_dim, y_dim, num_shots):
    # returns the measured image as a numpy array
    
    pixel_dim = 2 ** pixel_depth
    meas_state_idx = 0
    x_bsize = int(np.log2(x_dim))
    y_bsize = int(np.log2(y_dim))
    reconst_img_arr = np.zeros((x_dim, y_dim))
    # Probability density function of states
    probs = np.power(np.absolute(enc_input), 2)

    # Cumulative distribution function
    cdf = np.zeros((np.size(probs)))
    for i in range(np.size(cdf)):
        if(i == 0):
            cdf[i] = probs[i]
        else:
            cdf[i] = probs[i] + cdf[i-1]
          
    # Measurements
    for n in range(num_shots):
        r = random.random()
        for i in range(np.size(cdf)):
            if(i != np.size(cdf)-1):
                if((r > cdf[i]) and (r < cdf[i+1])):
                    meas_state_idx = i+1
      
        # inferring the pixel value and position
        meas_state_bin = format(meas_state_idx, '08b')
        # split into x, y and pixel
        x_str = meas_state_bin[-1*x_bsize:]
        y_str = meas_state_bin[-1*(x_bsize + y_bsize):-1*x_bsize]
        pixel_str = meas_state_bin[0:-1*(x_bsize + y_bsize)]
        x_loc = int(x_str, 2)
        y_loc = int(y_str, 2)
        pixel_int = int(pixel_str, 2)
        reconst_img_arr[x_loc, y_loc] = pixel_int
        
    return reconst_img_arr.astype(int)