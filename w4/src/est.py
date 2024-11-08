import cv2
from skimage.restoration import estimate_sigma
import os
import math
from scipy.signal import convolve2d
import numpy as np

def estimate_noise(I):

  H, W = I.shape

  M = [[1, -2, 1],
       [-2, 4, -2],
       [1, -2, 1]]

  sigma = np.sum(np.sum(np.absolute(convolve2d(I, M))))
  sigma = sigma * math.sqrt(0.5 * math.pi) / (6 * (W-2) * (H-2))

  return sigma

if __name__ == "__main__":
    for image_path in os.listdir('./data/qsd1_w4'):
        if image_path.endswith('.jpg'):

            img = cv2.imread('./data/qsd1_w4/'+image_path)
            grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            noiseLevel = estimate_noise(grayImage)
            if noiseLevel < 10:
                print("Noise Level of ",image_path," : ", "Image is clean")
            else:
                print("Noise Level of ",image_path," : ", "Image NOT clean")