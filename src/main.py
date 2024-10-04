import os
import cv2
import numpy as np
from descriptors import ImageDescriptor
import pickle

def process_images(folder_path, descriptor):
    histograms_dict = {}
    
    # Iterar sobre todos los archivos en la carpeta
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg"):  # Filtrar imágenes por extensión
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)
            
            if image is None:
                print(f"Error loading image {image_path}")
                continue
            
            # Get the descriptor histogram
            hist = descriptor.describe(image)
            
            # Save the histogram as an image
            output_image_path = os.path.join('results', f'{filename.split(".")[0]}_{descriptor.color_space.lower()}_hist.png')
            descriptor.save_histogram(hist, output_image_path)
            
            # Save the histogram in the npy dictionary
            histograms_dict[filename] = {
                'color_space': descriptor.color_space,
                'histogram_bins': descriptor.histogram_bins,
                'histograms': hist
            }
    
    return histograms_dict

def mAPK(K, hist, labels, similarity_measure):

    top_K = []

    if similarity_measure == "intersection":
        pass
    else:
        pass

    return top_K


if __name__ == '__main__':
    folder_path = './data/BBDD'
    results_folder = './results'

    # Check if .npy files exist
    cielab_npy = os.path.join(results_folder, 'histograms_cielab.npy')
    hsv_npy = os.path.join(results_folder, 'histograms_hsv.npy')

    # Create results folder if it doesn't exist
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    # Check if histograms have already been calculated
    if os.path.exists(cielab_npy):
        print(f"Loading CIELAB histograms from {cielab_npy}...")
        histograms_cielab = np.load(cielab_npy, allow_pickle=True).item()
    else:
        print("Calculating CIELAB histograms...")
        cielab_descriptor = ImageDescriptor('CIELAB')
        histograms_cielab = process_images(folder_path, cielab_descriptor)
        np.save(cielab_npy, histograms_cielab)
        print(f"CIELAB histograms saved as {cielab_npy}")

    if os.path.exists(hsv_npy):
        print(f"Loading HSV histograms from {hsv_npy}...")
        histograms_hsv = np.load(hsv_npy, allow_pickle=True).item()
    else:
        print("Calculating HSV histograms...")
        hsv_descriptor = ImageDescriptor('HSV')
        histograms_hsv = process_images(folder_path, hsv_descriptor)
        np.save(hsv_npy, histograms_hsv)
        print(f"HSV histograms saved as {hsv_npy}")

    print("All histograms have been successfully loaded or calculated.")

    # Load the labels for the comparison

    with open('..\data\qsd1_w1\gt_corresps.pkl') as f:
        labels = pickle.load(f)

    # Calculate similarities with CIELAB histograms
    # First, using histogram Intersection

    mapInterCIELAB_1 = mAPK(1, histograms_cielab, labels, "intersection")
    mapInterCIELAB_5 = mAPK(5, histograms_cielab, labels, "intersection")

    # Second, using Bhattacharyya distance

    mapBhattCIELAB_1 = mAPK(1, histograms_cielab, labels, "bhatt")
    mapBhattCIELAB_5 = mAPK(5, histograms_cielab, labels, "bhatt")

    # Calculate similarities with HSV histograms
    # First, using histogram Intersection

    mapInterHSV_1 = mAPK(1, histograms_hsv, labels, "intersection")
    mapInterHSV_5 = mAPK(5, histograms_hsv, labels, "intersection")

    # Second, using Bhattacharyya distance

    mapBhattHSV_1 = mAPK(1, histograms_hsv, labels, "bhatt")
    mapBhattHSV_5 = mAPK(5, histograms_hsv, labels, "bhatt")
