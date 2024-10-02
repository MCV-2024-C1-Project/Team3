# descriptors.py - Task 1
import cv2
import matplotlib.pyplot as plt
import numpy as np

class ImageDescriptor:

    def __init__(self, color_space, histogram_bins=[8, 8, 8]):  # default value for histogram_bins [R, G, B]
        self.color_space = color_space
        self.histogram_bins = histogram_bins


    def describe(self, image):
        # select color space
        if self.color_space == 'CIELAB':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            range = [0, 256, 0, 256, 0, 256]
        elif self.color_space == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            range = [0, 180, 0, 256, 0, 256]
        else:
            raise ValueError("Unsupported color space: {}".format(self.color_space))

        # compute histogram
        hist = cv2.calcHist([image], [0, 1, 2], None, self.histogram_bins, range)
        hist = cv2.normalize(hist, hist).flatten()

        return hist
    

    def save_histogram(self, hist, output_path):
        plt.figure()
        plt.bar(np.arange(len(hist)), hist, color='b', edgecolor='black')   # TODO: change the colors of the bars for a better visualization
        plt.title(f"Histogram ({self.color_space} space)")
        plt.xlabel("Bin")
        plt.ylabel("Frequency")
        
        plt.savefig(output_path)
        plt.close() 
        