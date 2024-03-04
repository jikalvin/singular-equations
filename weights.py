import numpy as np

def weight_function_first_kind(x):
    return 1 / np.sqrt(1 - x**2)

def weight_function_second_kind(x):
    return 1

def weight_function_third_kind(x):
    return np.sqrt(1 - x**2)

def weight_function_fourth_kind(x):
    return np.sqrt(1 - x**2)
