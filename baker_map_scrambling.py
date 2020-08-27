#!/usr/bin/env python
# coding: utf-8

import numpy as np
import sys
import seaborn as sns
import matplotlib.pyplot as plt

# Some helper functions

def is_swap_condition_true(in_idx, swap_idx_1, swap_idx_2, ctrl_idx, x_dim, y_dim, pixel_depth):
    # swap_idx_1 and swap_idx_2 are qubit indices appearing in the NEQR image state, not in terms of x and y
    # ctrl_idx is to be supplied as a value > -1 to implement single-qubit controlled swap
    max_idx = x_dim * y_dim * (2 ** pixel_depth)
    if((in_idx > max_idx) or (swap_idx_1 > max_idx) or (swap_idx_2 > max_idx) or (ctrl_idx > max_idx)):
        print("ERROR", "is_swap_condition_true")
        return
    in_idx_str = format(in_idx, '0'+str(pixel_depth + int(np.log2(x_dim)) + int(np.log2(y_dim)))+'b')
    if(ctrl_idx < 0):
        if(in_idx_str[swap_idx_1] != in_idx_str[swap_idx_2]):
            return True
        else:
            return False
    else:
        # Controlled Swap
        if((in_idx_str[ctrl_idx] == '1') and (in_idx_str[swap_idx_1] != in_idx_str[swap_idx_2])):
            return True
        else:
            return False

def get_out_idx_swap(in_idx, swap_idx_1, swap_idx_2, x_dim, y_dim, pixel_depth):
    # swap_idx_1 and swap_idx_2 are absolute matrix indices, not in terms of x and y
    max_idx = x_dim * y_dim * (2 ** pixel_depth)
    if((in_idx > max_idx) or (swap_idx_1 > max_idx) or (swap_idx_2 > max_idx)):
        print("ERROR", "get_out_idx_swap")
        return
    in_idx_str = format(in_idx, '0'+str(pixel_depth + int(np.log2(x_dim)) + int(np.log2(y_dim)))+'b')
    temp = in_idx_str[swap_idx_1]
    in_idx_str = in_idx_str[:swap_idx_1] + in_idx_str[swap_idx_2] + in_idx_str[swap_idx_1 + 1:]
    in_idx_str = in_idx_str[:swap_idx_2] + temp + in_idx_str[swap_idx_2 + 1:]
    return int(in_idx_str, 2)

def get_x_relative_idx(x, x_dim, y_dim, pixel_depth):
    # convention: index 0 is MSB
    # x (y) varies from 0 to np.log2(x_dim) - 1 (np.log2(y_dim) - 1)
    max_idx = x_dim * y_dim * (2 ** pixel_depth)
    if((x > max_idx) or (x > int(np.log2(x_dim)))):
        print("ERROR", "get_x_relative_idx", "x = ", x, " max_idx = ", max_idx, " int(np.log2(x_dim)) = ", int(np.log2(x_dim)))
        return
    return pixel_depth + x

def get_y_relative_idx(y, x_dim, y_dim, pixel_depth):
    # convention: index 0 is MSB
    # x (y) varies from 0 to np.log2(x_dim) - 1 (np.log2(y_dim) - 1)
    max_idx = x_dim * y_dim * (2 ** pixel_depth)
    if((y > max_idx) or (y > int(np.log2(y_dim)))):
        print("ERROR", "get_y_relative_idx")
        return
    return pixel_depth + int(np.log2(x_dim)) + y

def init_img_gate(x_dim, y_dim, pixel_depth):
    from numpy import zeros
    max_dim = x_dim * y_dim * (2 ** pixel_depth)
    #return np.zeros((max_dim, max_dim)) + np.zeros((max_dim, max_dim))*1.0j
    return zeros((max_dim, max_dim), dtype=np.uint8)

def get_2_qubit_swap_gate(swap_idx_1, swap_idx_2, ctrl_idx, x_dim, y_dim, pixel_depth):
    # pixel depth is depth of individual pixel in bits
    max_idx = x_dim * y_dim * (2 ** pixel_depth)
    if((swap_idx_1 > max_idx) or (swap_idx_2 > max_idx) or (ctrl_idx > max_idx)):
        print("ERROR", "get_2_qubit_swap_gate", " swap_idx_1 = ", swap_idx_1, " swap_idx_2 = ", swap_idx_2, " ctrl_idx = ", ctrl_idx, " max_idx = ", max_idx)
        return
    gate_mtx = init_img_gate(x_dim, y_dim, pixel_depth)
    for swap_in_idx in range(max_idx):
        is_swap = is_swap_condition_true(swap_in_idx, swap_idx_1, swap_idx_2, ctrl_idx, x_dim, y_dim, pixel_depth)
        swap_out_idx = get_out_idx_swap(swap_in_idx, swap_idx_1, swap_idx_2, x_dim, y_dim, pixel_depth)
        if(is_swap):
            gate_mtx[swap_out_idx, swap_in_idx] = 1;
        else:
            gate_mtx[swap_in_idx, swap_in_idx] = 1;
    return gate_mtx

def get_baker_scrambling_gate(x_dim, y_dim, pixel_depth):
    n_x = int(np.log2(x_dim))
    n_y = int(np.log2(y_dim))
    # x_n-2 swap y_0
    G0 = get_2_qubit_swap_gate(get_x_relative_idx(n_x-2, x_dim, y_dim, pixel_depth), get_y_relative_idx(0, x_dim, y_dim, pixel_depth), -1, x_dim, y_dim, pixel_depth)
    # x_n-1 swap y_1
    G0 = np.matmul(get_2_qubit_swap_gate(get_x_relative_idx(n_x-1, x_dim, y_dim, pixel_depth), get_y_relative_idx(1, x_dim, y_dim, pixel_depth), -1, x_dim, y_dim, pixel_depth), G0)
    # y_j swap y_j+2 # Goes upto y_n-3
    for j in range(n_y-2):
        G0 = np.matmul(get_2_qubit_swap_gate(get_y_relative_idx(j, x_dim, y_dim, pixel_depth), get_y_relative_idx(j+2, x_dim, y_dim, pixel_depth), -1, x_dim, y_dim, pixel_depth), G0)
    # x_i swap x_i+2 # Goes upto x_n-3
    for i in range(n_x-2):
        G0 = np.matmul(get_2_qubit_swap_gate(get_x_relative_idx(n_x-1-i, x_dim, y_dim, pixel_depth), get_x_relative_idx(n_x-3-i, x_dim, y_dim, pixel_depth), -1, x_dim, y_dim, pixel_depth), G0)    
    # controlled y_j swap y_j+1
    for j in range(n_y-2):
        G0 = np.matmul(get_2_qubit_swap_gate(get_y_relative_idx(n_y-2-j, x_dim, y_dim, pixel_depth), get_y_relative_idx(n_y-3-j, x_dim, y_dim, pixel_depth), get_y_relative_idx(n_y-1, x_dim, y_dim, pixel_depth), x_dim, y_dim, pixel_depth), G0) 
    # controlled x_i swap x_i+1
    for i in range(n_x-2):
        G0 = np.matmul(get_2_qubit_swap_gate(get_x_relative_idx(i+1, x_dim, y_dim, pixel_depth), get_x_relative_idx(i+2, x_dim, y_dim, pixel_depth), get_y_relative_idx(n_y-1, x_dim, y_dim, pixel_depth),x_dim, y_dim, pixel_depth), G0)      
    return G0

