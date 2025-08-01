import numpy as np

def get_class_centers(X, Y, class_list):
    return {label: X[Y == label].mean(axis=0) for label in class_list}