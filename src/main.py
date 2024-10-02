import cv2
from descriptors import ImageDescriptor

if __name__ == '__main__':
    cielab = ImageDescriptor('CIELAB')
    hsv = ImageDescriptor('HSV')

    # Load image and compute histograms
    image = cv2.imread('./data/BBDD/bbdd_00000.jpg')
    hist = cielab.describe(image)
    cielab.save_histogram(hist, 'results/cielab_hist.png')


