# descriptors.py - Task 1
import cv2
import matplotlib.pyplot as plt

class ImageDescriptor:

    def __init__(self, color_space, range, histogram_bins=255):
        self.color_space = color_space
        self.histogram_bins = histogram_bins
        self.range = range

    def describe(self, image):
        # select color space
        if self.color_space == 'CIELAB':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            self.range = [0, 256, 0, 256, 0, 256]
        elif self.color_space == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            self.range = [0, 180, 0, 256, 0, 256]
        else:
            # error
            pass

        # compute histogram
        hist = cv2.calcHist([image], [0, 1, 2], None, self.histogram_bins, self.range)
        hist = cv2.normalize(hist, hist).flatten()

        return hist

    def save_histogram(self, hist, output_path):
        plt.bar(range(len(hist)), hist, color='b', edgecolor='black')
        plt.savefig(output_path)
        