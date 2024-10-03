# descriptors.py - Task 1
import cv2
import matplotlib.pyplot as plt
import numpy as np

class ImageDescriptor:
    # Define the ranges for each color space
    COLOR_RANGES = {
        'CIELAB': [[0, 256], [0, 256], [0, 256]],
        'HSV': [[0, 180], [0, 256], [0, 256]]
    }

    # Define the names for each channel
    CHANNEL_NAMES = {
        'CIELAB': ['L*', 'a*', 'b*'],
        'HSV': ['H', 'S', 'V']
    }

    def __init__(self, color_space, histogram_bins=[256, 256, 256]):  # default value for histogram_bins [R, G, B]
        self.color_space = color_space
        self.histogram_bins = histogram_bins
        if color_space not in self.COLOR_RANGES:
            raise ValueError(f"Unsupported color space: {color_space}")


    def describe(self, image):
        # Select color space
        if self.color_space == 'CIELAB':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        elif self.color_space == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define the ranges for each channel
        ranges = self.COLOR_RANGES[self.color_space]

        histograms = []

        # Calculate histogram for each channel
        for i in range(3):
            hist = cv2.calcHist([image], [i], None, [self.histogram_bins[i]], ranges[i])
            histograms.append(hist.squeeze())

        return histograms
    

    def save_histogram(self, hist, output_path):
        #Create figure with 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.tight_layout(h_pad=2,w_pad=5) #ensure no overlapping between subplot

        channel_names = self.CHANNEL_NAMES[self.color_space]

        #Plot each histogram in a barplot
        for i, ax in enumerate(axes):
            ax.bar(np.arange(len(hist[i])), hist[i], color=['b', 'g', 'r'][i], edgecolor=['b', 'g', 'r'][i])
            ax.set_title(f'{channel_names[i]}')
        
        fig.suptitle(f'{self.color_space} Histogram')

        plt.subplots_adjust(top=0.85) #Adjust spacing of figure title

        fig.savefig(output_path,bbox_inches='tight') #store image
        