import numpy as np

def calculate_accuracy(predictions, labels):
    return np.mean(predictions == labels)