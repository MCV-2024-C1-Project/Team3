# descriptors.py - Task 1
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import dct
from skimage.transform import rotate
from skimage.feature import local_binary_pattern
from skimage import data
from skimage.color import label2rgb

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

    def describe(self, image, dimension, structure, mask=None):
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
                            h1 = cv2.calcHist([image1], [0, 1, 2], mask, [16, 16, 16], [0, 256, 0, 256, 0, 256])
                            h2 = cv2.calcHist([image2], [0, 1, 2], mask, [16, 16, 16], [0, 256, 0, 256, 0, 256])
                            h3 = cv2.calcHist([image3], [0, 1, 2], mask, [16, 16, 16], [0, 256, 0, 256, 0, 256])
                            hist.append(h1.flatten())
                            hist.append(h2.flatten())
                            hist.append(h3.flatten())
                        elif dimension == '2D':
                            for i in range(3):
                                h1 = cv2.calcHist([image1], [i], mask, [self.histogram_bins[i]], ranges[i])
                                h2 = cv2.calcHist([image2], [i], mask, [self.histogram_bins[i]], ranges[i])
                                h3 = cv2.calcHist([image3], [i], mask, [self.histogram_bins[i]], ranges[i])
                                hist.append(h1.flatten())
                                hist.append(h2.flatten())
                                hist.append(h3.flatten())
                    else:
                        subimgs = self.block(image1, 2**b)
                        for img in subimgs:
                            if dimension == '3D':
                                h1 = cv2.calcHist([img], [0, 1, 2], mask, [16, 16, 16], [0, 256, 0, 256, 0, 256])
                                hist.append(h1.flatten())
                            elif dimension == '2D':
                                for i in range(3):
                                    h1 = cv2.calcHist([img], [i], mask, [self.histogram_bins[i]], ranges[i])
                                    hist.append(h1.flatten())

                        subimgs = self.block(image2, 2**b)
                        for img in subimgs:
                            if dimension == '3D':
                                h2 = cv2.calcHist([img], [0, 1, 2], mask, [16, 16, 16], [0, 256, 0, 256, 0, 256])
                                hist.append(h2.flatten())

                            elif dimension == '2D':
                                for i in range(3):
                                    h2  = cv2.calcHist([img], [i], mask, [self.histogram_bins[i]], ranges[i])
                                    hist.append(h2.flatten())

                        subimgs = self.block(image3, 2**b)
                        for img in subimgs:
                            if dimension == '3D':
                                h3 = cv2.calcHist([img], [0, 1, 2], mask, [16, 16, 16], [0, 256, 0, 256, 0, 256])
                                hist.append(h3.flatten())    

                            elif dimension == '2D':
                                for i in range(3):
                                    h3 = cv2.calcHist([img], [i], mask, [self.histogram_bins[i]], ranges[i])
                                    hist.append(h3.flatten())
                    histograms[b] = {'histogram': np.concatenate(hist).flatten()}
                return histograms
            elif structure == 'heriarchical':
                histograms = {}
                hist = []
                for b in range(3):
                    if b == 0:
                        if dimension == '3D':
                            h1 = cv2.calcHist([image1], [0, 1, 2], mask, [16, 16, 16], [0, 256, 0, 256, 0, 256])
                            h2 = cv2.calcHist([image2], [0, 1, 2], mask, [16, 16, 16], [0, 256, 0, 256, 0, 256])
                            h3 = cv2.calcHist([image3], [0, 1, 2], mask, [16, 16, 16], [0, 256, 0, 256, 0, 256])
                            hist.append(h1.flatten())
                            hist.append(h2.flatten())
                            hist.append(h3.flatten())
                        elif dimension == '2D':
                            for i in range(3):
                                h1 = cv2.calcHist([image1], [i], mask, [self.histogram_bins[i]], ranges[i])
                                h2 = cv2.calcHist([image2], [i], mask, [self.histogram_bins[i]], ranges[i])
                                h3 = cv2.calcHist([image3], [i], mask, [self.histogram_bins[i]], ranges[i])
                                hist.append(h1.flatten())
                                hist.append(h2.flatten())
                                hist.append(h3.flatten())
                    else:
                        subimgs = self.block(image1, 2**b)
                        for img in subimgs:
                            if dimension == '3D':
                                h1 = cv2.calcHist([img], [0, 1, 2], mask, [16, 16, 16], [0, 256, 0, 256, 0, 256])
                                hist.append(h1.flatten())
                            elif dimension == '2D':
                                for i in range(3):
                                    h1 = cv2.calcHist([img], [i], mask, [self.histogram_bins[i]], ranges[i])
                                    hist.append(h1.flatten())

                        subimgs = self.block(image2, 2**b)
                        for img in subimgs:
                            if dimension == '3D':
                                h2 = cv2.calcHist([img], [0, 1, 2], mask, [16, 16, 16], [0, 256, 0, 256, 0, 256])
                                hist.append(h2.flatten())

                            elif dimension == '2D':
                                for i in range(3):
                                    h2  = cv2.calcHist([img], [i], mask, [self.histogram_bins[i]], ranges[i])
                                    hist.append(h2.flatten())

                        subimgs = self.block(image3, 2**b)
                        for img in subimgs:
                            if dimension == '3D':
                                h3 = cv2.calcHist([img], [0, 1, 2], mask, [16, 16, 16], [0, 256, 0, 256, 0, 256])
                                hist.append(h3.flatten())    

                            elif dimension == '2D':
                                for i in range(3):
                                    h3 = cv2.calcHist([img], [i], mask, [self.histogram_bins[i]], ranges[i])
                                    hist.append(h3.flatten())
                    histograms[b] = {'histogram': np.concatenate(hist).flatten()}
                return histograms
        else: 
            if structure == 'simple':
                histograms = []
                if dimension == '3D':
                    hist = cv2.calcHist([image], [0, 1, 2], mask, [128, 128, 128], [0, 256, 0, 256, 0, 256])
                    histograms.append(hist.flatten())
                elif dimension == '2D':
                    for i in range(3):
                        hist = cv2.calcHist([image], [i], mask, [self.histogram_bins[i]], ranges[i])
                        histograms.append(hist.flatten())
                return np.concatenate(histograms).flatten()

            elif structure == 'block' :
                histograms = {}
                for b in range(3):
                    hist = []
                    if b == 0:
                        if dimension == '3D':
                            h = cv2.calcHist([image], [0, 1, 2], mask, [16, 16, 16], [0, 256, 0, 256, 0, 256])
                            hist.append(h.flatten())
                        elif dimension == '2D':
                            for i in range(3):
                                h = cv2.calcHist([image], [i], mask, [self.histogram_bins[i]], ranges[i])
                                hist.append(h.flatten())
                    else:
                        subimgs = self.block(image, 2**b)
                        for img in subimgs:
                            if dimension == '3D':
                                h = cv2.calcHist([img], [0, 1, 2], mask, [16, 16, 16], [0, 256, 0, 256, 0, 256])
                                hist.append(h.flatten())
                            elif dimension == '2D':
                                for i in range(3):
                                    h = cv2.calcHist([img], [i], mask, [self.histogram_bins[i]], ranges[i])
                                    hist.append(h.flatten())
                    histograms[b] = {'histogram': np.concatenate(hist).flatten()}
                return histograms
            
            elif structure == 'heriarchical':
                histograms = {}
                hist = []
                for b in range(3):
                    if b == 0:
                        if dimension == '3D':
                            h = cv2.calcHist([image], [0, 1, 2], mask, [16, 16, 16], [0, 256, 0, 256, 0, 256])
                            hist.append(h.flatten())
                        elif dimension == '2D':
                            for i in range(3):
                                h = cv2.calcHist([image], [i], mask, [self.histogram_bins[i]], ranges[i])
                                hist.append(h.flatten())
                    else:
                        subimgs = self.block(image, 2**b)
                        for img in subimgs:
                            if dimension == '3D':
                                h = cv2.calcHist([img], [0, 1, 2], mask, [16, 16, 16], [0, 256, 0, 256, 0, 256])
                                hist.append(h.flatten())
                            elif dimension == '2D':
                                for i in range(3):
                                    h = cv2.calcHist([img], [i], mask, [self.histogram_bins[i]], ranges[i])
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

class TextureDescriptor:
    COLOR_RANGES = {
        'HLS': [[0, 256], [0, 256], [0, 256]],
        'HSV': [[0, 256], [0, 256], [0, 256]],
        'XYZ': [[0, 256], [0, 256], [0, 256]],
        'RGB': [[0, 256], [0, 256], [0, 256]],
        'LAB': [[0, 256], [0, 256], [0, 256]],
        'YCrCb': [[0, 256], [0, 256], [0, 256]],
        'combine':[[0, 256], [0, 256], [0, 256]],
        'gray':[0,256]
    }
    def __init__(self, color_space):

        self.color_space = color_space

        if color_space not in self.COLOR_RANGES:
            raise ValueError(f"Unsupported color space: {color_space}")

    def block(self, image, nblocks):
        divide1 = np.array_split(image, nblocks[0], axis=0)
        divide2 = [np.array_split(subimg, nblocks[1], axis=1) for subimg in divide1]
        subimgs = [img for row in divide2 for img in row]

        return subimgs
    
    def dctn(self,x, norm="ortho"):
        x=x-np.ones(np.shape(x))*128
        for i in range(x.ndim):
            x = dct(x, axis=i, norm=norm)
        return np.array(x)

    def quantize(self,dctresult):
        mat=np.array(
            [[16,11,10,16,24,40,51,61],
            [12,12,14,19,26,58,60,55],
            [14,13,16,24,40,57,69,56],
            [14,17,22,29,51,87,80,62],
            [18,22,37,56,68,109,103,77],
            [24,35,55,64,81,104,113,92],
            [49,64,78,87,103,121,120,101],
            [72,92,95,98,112,100,103,99]])
        return dctresult//mat
    

    def zigzag(self,input):
        # Zigzag scan of a matrix
        # Argument is a two-dimensional matrix of any size,
        # not strictly a square one.
        # Function returns a 1-by-(m*n) array,
        # where m and n are sizes of an input matrix,
        # consisting of its items scanned by a zigzag method.
        #
        # Matlab Code:
        # Alexey S. Sokolov a.k.a. nICKEL, Moscow, Russia
        # June 2007
        # alex.nickel@gmail.com
        
        
        #initializing the variables
        #----------------------------------
        h = 0
        v = 0

        vmin = 0
        hmin = 0

        vmax = input.shape[0]
        hmax = input.shape[1]
        
        #print(vmax ,hmax )

        i = 0

        output = np.zeros(( vmax * hmax))
        #----------------------------------

        while ((v < vmax) and (h < hmax)):
            
            if ((h + v) % 2) == 0:                 # going up
                
                if (v == vmin):
                    #print(1)
                    output[i] = input[v, h]        # if we got to the first line

                    if (h == hmax):
                        v = v + 1
                    else:
                        h = h + 1                        

                    i = i + 1

                elif ((h == hmax -1 ) and (v < vmax)):   # if we got to the last column
                    #print(2)
                    output[i] = input[v, h] 
                    v = v + 1
                    i = i + 1

                elif ((v > vmin) and (h < hmax -1 )):    # all other cases
                    #print(3)
                    output[i] = input[v, h] 
                    v = v - 1
                    h = h + 1
                    i = i + 1

            
            else:                                    # going down

                if ((v == vmax -1) and (h <= hmax -1)):       # if we got to the last line
                    #print(4)
                    output[i] = input[v, h] 
                    h = h + 1
                    i = i + 1
            
                elif (h == hmin):                  # if we got to the first column
                    #print(5)
                    output[i] = input[v, h] 

                    if (v == vmax -1):
                        h = h + 1
                    else:
                        v = v + 1

                    i = i + 1

                elif ((v < vmax -1) and (h > hmin)):     # all other cases
                    #print(6)
                    output[i] = input[v, h] 
                    v = v + 1
                    h = h - 1
                    i = i + 1




            if ((v == vmax-1) and (h == hmax-1)):          # bottom right element
                #print(7)        	
                output[i] = input[v, h] 
                break

        #print ('v:',v,', h:',h,', i:',i)
        return output

    def describe(self, image, structure, quantization):
        # image = cv2.medianBlur(image, 3)

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
        elif self.color_space=='gray':

            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
        # elif self.color_space=='combine':
        #     image1 = cv2.cvtColor(image, cv2.COLOR_BGR2HSV_FULL)
        #     image2 = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        #     image3 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if structure=='DCT':
            
            #image should be grayscale
            img=cv2.resize(image,(512,512))
            m,n=np.shape(img)
            coefs=[]
            divided_img=self.block(img,[m/8,n/8])
            for subimg in divided_img:
                dct_block=self.dctn(subimg)
                dct_coefs=self.zigzag(dct_block*255.0)
                coefs.append(dct_coefs[:10])
            return coefs

        elif structure=="LBP":
            n_points=8
            radius=2
            img=cv2.resize(image,(512,512))
            m,n=np.shape(img)
            histogram=[]
            divided_img=self.block(img,[m/8,n/8])
            for subimg in divided_img:
                lbp = np.float32(local_binary_pattern(subimg, n_points, radius, 'uniform'))
                hist= cv2.calcHist([lbp.flatten()],[0],None,[10],[0,10])/(m*n)
                histogram.append(hist)
        
            return histogram


                

            