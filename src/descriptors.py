# descriptors.py - Task 1
import cv2
import matplotlib.pyplot as plt
import numpy as np

class ImageDescriptor:

    def __init__(self, color_space, histogram_bins=[256, 256, 256]):  # default value for histogram_bins [R, G, B]
        self.color_space = color_space
        self.histogram_bins = histogram_bins


    def describe(self, image):
        # select color space
        if self.color_space == 'CIELAB':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            range = [[0, 256],[0, 256], [0, 256]]
            self.histogram_bins=[256,256,256]
        elif self.color_space == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            range = [[0, 180], [0, 256], [0, 256]]
            self.histogram_bins=[180,256,256]
        else:
            raise ValueError("Unsupported color space: {}".format(self.color_space))

        # compute histogram separatedly for each channel
        a = cv2.calcHist([image], [0], None, [self.histogram_bins[0]], range[0])
        b = cv2.calcHist([image], [1], None, [self.histogram_bins[1]], range[1])
        c = cv2.calcHist([image], [2], None, [self.histogram_bins[2]], range[2])

        #Join histograms
        hist=[]
        hist.append(a.squeeze())
        hist.append(b.squeeze())
        hist.append(c.squeeze())

        return hist
    

    def save_histogram(self, hist, output_path):
        #Create figure with 3 subplots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        fig.tight_layout(h_pad=2,w_pad=5) #ensure no overlapping between subplot

        #Plot each histogram in a barplot
        ax1.bar(np.arange(np.shape(hist)[1]), hist[0], color='b', edgecolor='b')   
        ax2.bar(np.arange(np.shape(hist)[1]), hist[1], color='g', edgecolor='g')   
        ax3.bar(np.arange(np.shape(hist)[1]), hist[2], color='r', edgecolor='r')   
        
        #Define subtitles for each subplot and figure title
        if 'cielab' in output_path:
            ax1.set_title('L*')
            ax2.set_title('a*')
            ax3.set_title('b*')
            fig.suptitle('CIELAB histogram')

        elif 'HSV' in output_path:
            ax1.set_title('H')
            ax2.set_title('S')
            ax3.set_title('V')
            fig.suptitle('HSV histogram')
        
        plt.subplots_adjust(top=0.85) #Adjust spacing of figure title

        fig.savefig(output_path,bbox_inches='tight') #store image
        