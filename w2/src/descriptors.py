# descriptors.py - Task 1
import cv2
import matplotlib.pyplot as plt
import numpy as np

class ImageDescriptor:
    COLOR_RANGES = {
        'HLS': [[0, 256], [0, 256], [0, 256]],
        'HSV': [[0, 256], [0, 256], [0, 256]],
        'XYZ': [[0, 256], [0, 256], [0, 256]],
        'RGB': [[0, 256], [0, 256], [0, 256]],
        'LAB': [[0, 256], [0, 256], [0, 256]],
        'YCrCb': [[0, 256], [0, 256], [0, 256]],
        'combine':[[0, 256], [0, 256], [0, 256]]
    }

    CHANNEL_NAMES = {
        'HLS': ['H', 'L', 'S'],
        'HSV': ['H', 'S', 'V']
    }

    def __init__(self, color_space, histogram_bins=[256, 256, 256]):
        self.color_space = color_space
        self.histogram_bins = histogram_bins
        if color_space not in self.COLOR_RANGES:
            raise ValueError(f"Unsupported color space: {color_space}")

    def block(self, image, nblocks):
        divide1 = np.array_split(image, nblocks, axis=0)
        divide2 = [np.array_split(subimg, nblocks, axis=1) for subimg in divide1]
        subimgs = [img for row in divide2 for img in row]
        return subimgs

    def describe(self, image, dimension, structure):
        if self.color_space == 'HLS':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS_FULL)
        elif self.color_space == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV_FULL)
        elif self.color_space == 'XYZ':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2XYZ)
        elif self.color_space == 'RGB':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif self.color_space == 'LAB':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        elif self.color_space == 'YCrCb':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)   
        elif self.color_space=='combine':
            image1 = cv2.cvtColor(image, cv2.COLOR_BGR2HSV_FULL)
            image2 = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            image3 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)



        ranges = self.COLOR_RANGES[self.color_space]
        if self.color_space=='combine':

            if structure == 'block' :
                histograms = {}
                for b in range(3):
                    hist = []
                    if b == 0:
                        if dimension == '3D':
                            h1 = cv2.calcHist([image1], [0, 1, 2], None, [16, 16, 16], [0, 256, 0, 256, 0, 256])
                            h2 = cv2.calcHist([image2], [0, 1, 2], None, [16, 16, 16], [0, 256, 0, 256, 0, 256])
                            h3 = cv2.calcHist([image3], [0, 1, 2], None, [16, 16, 16], [0, 256, 0, 256, 0, 256])
                            hist.append(h1.flatten())
                            hist.append(h2.flatten())
                            hist.append(h3.flatten())
                        elif dimension == '2D':
                            for i in range(3):
                                h1 = cv2.calcHist([image1], [i], None, [self.histogram_bins[i]], ranges[i])
                                h2 = cv2.calcHist([image2], [i], None, [self.histogram_bins[i]], ranges[i])
                                h3 = cv2.calcHist([image3], [i], None, [self.histogram_bins[i]], ranges[i])
                                hist.append(h1.flatten())
                                hist.append(h2.flatten())
                                hist.append(h3.flatten())
                    else:
                        subimgs = self.block(image1, 2**b)
                        for img in subimgs:
                            if dimension == '3D':
                                h1 = cv2.calcHist([img], [0, 1, 2], None, [16, 16, 16], [0, 256, 0, 256, 0, 256])
                                hist.append(h1.flatten())
                            elif dimension == '2D':
                                for i in range(3):
                                    h1 = cv2.calcHist([img], [i], None, [self.histogram_bins[i]], ranges[i])
                                    hist.append(h1.flatten())

                        subimgs = self.block(image2, 2**b)
                        for img in subimgs:
                            if dimension == '3D':
                                h2 = cv2.calcHist([img], [0, 1, 2], None, [16, 16, 16], [0, 256, 0, 256, 0, 256])
                                hist.append(h2.flatten())

                            elif dimension == '2D':
                                for i in range(3):
                                    h2  = cv2.calcHist([img], [i], None, [self.histogram_bins[i]], ranges[i])
                                    hist.append(h2.flatten())

                        subimgs = self.block(image3, 2**b)
                        for img in subimgs:
                            if dimension == '3D':
                                h3 = cv2.calcHist([img], [0, 1, 2], None, [16, 16, 16], [0, 256, 0, 256, 0, 256])
                                hist.append(h3.flatten())    

                            elif dimension == '2D':
                                for i in range(3):
                                    h3 = cv2.calcHist([img], [i], None, [self.histogram_bins[i]], ranges[i])
                                    hist.append(h3.flatten())
                    histograms[b] = {'histogram': np.concatenate(hist).flatten()}
                return histograms
            elif structure == 'heriarchical':
                histograms = {}
                hist = []
                for b in range(3):
                    if b == 0:
                        if dimension == '3D':
                            h1 = cv2.calcHist([image1], [0, 1, 2], None, [16, 16, 16], [0, 256, 0, 256, 0, 256])
                            h2 = cv2.calcHist([image2], [0, 1, 2], None, [16, 16, 16], [0, 256, 0, 256, 0, 256])
                            h3 = cv2.calcHist([image3], [0, 1, 2], None, [16, 16, 16], [0, 256, 0, 256, 0, 256])
                            hist.append(h1.flatten())
                            hist.append(h2.flatten())
                            hist.append(h3.flatten())
                        elif dimension == '2D':
                            for i in range(3):
                                h1 = cv2.calcHist([image1], [i], None, [self.histogram_bins[i]], ranges[i])
                                h2 = cv2.calcHist([image2], [i], None, [self.histogram_bins[i]], ranges[i])
                                h3 = cv2.calcHist([image3], [i], None, [self.histogram_bins[i]], ranges[i])
                                hist.append(h1.flatten())
                                hist.append(h2.flatten())
                                hist.append(h3.flatten())
                    else:
                        subimgs = self.block(image1, 2**b)
                        for img in subimgs:
                            if dimension == '3D':
                                h1 = cv2.calcHist([img], [0, 1, 2], None, [16, 16, 16], [0, 256, 0, 256, 0, 256])
                                hist.append(h1.flatten())
                            elif dimension == '2D':
                                for i in range(3):
                                    h1 = cv2.calcHist([img], [i], None, [self.histogram_bins[i]], ranges[i])
                                    hist.append(h1.flatten())

                        subimgs = self.block(image2, 2**b)
                        for img in subimgs:
                            if dimension == '3D':
                                h2 = cv2.calcHist([img], [0, 1, 2], None, [16, 16, 16], [0, 256, 0, 256, 0, 256])
                                hist.append(h2.flatten())

                            elif dimension == '2D':
                                for i in range(3):
                                    h2  = cv2.calcHist([img], [i], None, [self.histogram_bins[i]], ranges[i])
                                    hist.append(h2.flatten())

                        subimgs = self.block(image3, 2**b)
                        for img in subimgs:
                            if dimension == '3D':
                                h3 = cv2.calcHist([img], [0, 1, 2], None, [16, 16, 16], [0, 256, 0, 256, 0, 256])
                                hist.append(h3.flatten())    

                            elif dimension == '2D':
                                for i in range(3):
                                    h3 = cv2.calcHist([img], [i], None, [self.histogram_bins[i]], ranges[i])
                                    hist.append(h3.flatten())
                    histograms[b] = {'histogram': np.concatenate(hist).flatten()}
                return histograms

        if structure == 'simple':
            histograms = []
            if dimension == '3D':
                hist = cv2.calcHist([image], [0, 1, 2], None, [128, 128, 128], [0, 256, 0, 256, 0, 256])
                histograms.append(hist.flatten())
            elif dimension == '2D':
                for i in range(3):
                    hist = cv2.calcHist([image], [i], None, [self.histogram_bins[i]], ranges[i])
                    histograms.append(hist.flatten())
            return np.concatenate(histograms).flatten()

        elif structure == 'block' :
            histograms = {}
            for b in range(3):
                hist = []
                if b == 0:
                    if dimension == '3D':
                        h = cv2.calcHist([image], [0, 1, 2], None, [16, 16, 16], [0, 256, 0, 256, 0, 256])
                        hist.append(h.flatten())
                    elif dimension == '2D':
                        for i in range(3):
                            h = cv2.calcHist([image], [i], None, [self.histogram_bins[i]], ranges[i])
                            hist.append(h.flatten())
                else:
                    subimgs = self.block(image, 2**b)
                    for img in subimgs:
                        if dimension == '3D':
                            h = cv2.calcHist([img], [0, 1, 2], None, [16, 16, 16], [0, 256, 0, 256, 0, 256])
                            hist.append(h.flatten())
                        elif dimension == '2D':
                            for i in range(3):
                                h = cv2.calcHist([img], [i], None, [self.histogram_bins[i]], ranges[i])
                                hist.append(h.flatten())
                histograms[b] = {'histogram': np.concatenate(hist).flatten()}
            return histograms
        elif structure == 'heriarchical':
            histograms = {}
            hist = []
            for b in range(3):
                if b == 0:
                    if dimension == '3D':
                        h = cv2.calcHist([image], [0, 1, 2], None, [16, 16, 16], [0, 256, 0, 256, 0, 256])
                        hist.append(h.flatten())
                    elif dimension == '2D':
                        for i in range(3):
                            h = cv2.calcHist([image], [i], None, [self.histogram_bins[i]], ranges[i])
                            hist.append(h.flatten())
                else:
                    subimgs = self.block(image, 2**b)
                    for img in subimgs:
                        if dimension == '3D':
                            h = cv2.calcHist([img], [0, 1, 2], None, [16, 16, 16], [0, 256, 0, 256, 0, 256])
                            hist.append(h.flatten())
                        elif dimension == '2D':
                            for i in range(3):
                                h = cv2.calcHist([img], [i], None, [self.histogram_bins[i]], ranges[i])
                                hist.append(h.flatten())
                histograms[b] = {'histogram': np.concatenate(hist).flatten()}
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

        # fig.savefig(output_path,bbox_inches='tight') #store image
        plt.close()  # Close the figure to free up memory