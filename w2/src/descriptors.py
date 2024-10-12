# descriptors.py - Task 1
import cv2
import matplotlib.pyplot as plt
import numpy as np

class ImageDescriptor:
    # Define the ranges for each color space
    COLOR_RANGES = {
        'HLS': [[0, 256], [0, 256], [0, 256]],
        'HSV': [[0, 256], [0, 256], [0, 256]]
    }

    # Define the names for each channel
    CHANNEL_NAMES = {
        'HLS': ['H', 'L', 'S'],
        'HSV': ['H', 'S', 'V']
    }

    def __init__(self, color_space, histogram_bins=[256, 256, 256]):  # default value for histogram_bins [R, G, B]
        self.color_space = color_space
        self.histogram_bins = histogram_bins
        if color_space not in self.COLOR_RANGES:
            raise ValueError(f"Unsupported color space: {color_space}")

    def block(self,image,nblocks):
        if np.shape(image)[0]%2:
            image=np.pad(image,((0,1),(0,0),(0,0)),'constant',constant_values=0)

        if np.shape(image)[0]%4==1:
            image=np.pad(image,((0,3),(0,0),(0,0)),'constant',constant_values=0)

        elif np.shape(image)[0]%4==2:
            image=np.pad(image,((0,2),(0,0),(0,0)),'constant',constant_values=0)
  
        elif np.shape(image)[0]%4==3:
            image=np.pad(image,((0,1),(0,0),(0,0)),'constant',constant_values=0)
        
        if np.shape(image)[1]%2:
            image=np.pad(image,((0,0),(0,1),(0,0)),'constant',constant_values=0)

        if np.shape(image)[1]%4==1:
            image=np.pad(image,((0,0),(0,3),(0,0)),'constant',constant_values=0)

        elif np.shape(image)[1]%4==2:
            image=np.pad(image,((0,0),(0,2),(0,0)),'constant',constant_values=0)
  
        elif np.shape(image)[1]%4==3:
            image=np.pad(image,((0,0),(0,1),(0,0)),'constant',constant_values=0)

        divide1=np.array_split(image,nblocks)
        divide2=[np.array_split(subimg,nblocks,axis=1) for subimg in divide1]
        return np.asarray(divide2, dtype=np.ndarray).reshape(nblocks*nblocks,np.shape(divide2)[2],np.shape(divide2)[3],3)


    def describe(self, image, dimension, structure):
        # Select color space
        if self.color_space == 'HLS':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS_FULL)
        elif self.color_space == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV_FULL)

        # Define the ranges for each channel
        ranges = self.COLOR_RANGES[self.color_space]
        
        if structure=='simple':
            histograms = []
            if dimension=='3D':
                hist = cv2.calcHist([image], [0,1,2], None, [128,128,128], [0, 256, 0, 256, 0, 256])
                histograms.append(hist.squeeze())
                histograms=np.array(histograms).squeeze()

            elif dimension=='2D':
                for i in range(3):
                    hist = cv2.calcHist([image], [i], None, [self.histogram_bins[i]], ranges[i])
                    histograms.append(hist.squeeze())

            else:
                raise ValueError(f"Unsupported dimension: {dimension}")
            histograms=np.array(histograms).flatten()
            
        elif structure=='block':
            histograms={}
            for b in range(3):
                hist=[]
                if b==0:
                    if dimension=='3D':
                        # print(np.shape(image))
                        h= cv2.calcHist([image], [0,1,2], None, [64,64,64], [0, 256, 0, 256, 0, 256])
                        hist.append(h)

                    elif dimension=='2D':
                        for i in range(3):
                            h = cv2.calcHist([image], [i], None, [self.histogram_bins[i]], ranges[i])
                            hist.append(h.squeeze())

                    else:
                        raise ValueError(f"Unsupported dimension: {dimension}")
                    
                else:

                    subimgs=self.block(image,2**b)
                    # subimgs=cv2.split(image)

                    for img in subimgs:
                        if dimension=='3D':
                            h = cv2.calcHist([img.astype(np.float32)], [0,1,2], None, [64,64,64], [0, 256, 0, 256, 0, 256])
                            hist.append(h)

                        elif dimension=='2D':
                            for i in range(3):
                                h = cv2.calcHist([image], [i], None, [self.histogram_bins[i]], ranges[i])
                                hist.append(h.squeeze())
    
                        else:
                            raise ValueError(f"Unsupported dimension: {dimension}")
                
               
                histograms[b] = {
                    'level': b,
                    'histogram': hist
                }
        
        
        elif structure=='heriarchical':
            histograms={}
            hist=[]
            for b in range(3):
                if b==0:
                    if dimension=='3D':
                        h= cv2.calcHist([image.astype(np.float32)], [0,1,2], None, [64,64,64], [0, 256, 0, 256, 0, 256])
                        hist.append(h)

                    elif dimension=='2D':
                        for i in range(3):
                            h = cv2.calcHist([image], [i], None, [self.histogram_bins[i]], ranges[i])
                            hist.append(h.squeeze())

                    else:
                        raise ValueError(f"Unsupported dimension: {dimension}")
                    
                else:

                    subimgs=self.block(image,2**b)

                    for img in subimgs:

                        if dimension=='3D':
                            h = cv2.calcHist([img.astype(np.float32)], [0,1,2], None, [64,64,64],  [0, 256, 0, 256, 0, 256])
                            hist.append(h)

                        elif dimension=='2D':
                            for i in range(3):
                                h = cv2.calcHist([image], [i], None, [self.histogram_bins[i]], ranges[i])
                                hist.append(h.squeeze())
    
                        else:
                            raise ValueError(f"Unsupported dimension: {dimension}")
                
               
                histograms[b] = {
                    'level': b,
                    'histogram': np.array(hist).flatten()
                }

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

        