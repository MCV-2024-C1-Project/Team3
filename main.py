import os
import cv2
import numpy as np
from src.descriptors import ImageDescriptor
import pickle
from evaluation.average_precision import mapk
from src.similarity import ComputeSimilarity

def process_name(name): 
    numbers = []

    for n in name:
        number_str = n[5:-4]  # "00060"

        # Convertir a entero para eliminar los ceros a la izquierda
        number = int(number_str)
        numbers.append(number)

    return numbers

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

def mAPK(K, hist, labels, similarity_measure, hist_type):

    top_K = []
    tot_top_K = []
    measures = ComputeSimilarity()

    for img in os.listdir('data/qsd1_w1'):
        if hist_type == "CIELAB":
            descriptor = ImageDescriptor('CIELAB')
            try:
                histogram = descriptor.describe(cv2.imread(f'data/qsd1_w1/{img}'))
            except: 
                continue
        else:
            descriptor = ImageDescriptor('HSV')
            try:
                histogram = descriptor.describe(cv2.imread(f'data/qsd1_w1/{img}'))
            except:
                continue
        
        if similarity_measure == "intersection":  
            similarities_inter = {key: measures.histogramIntersection(histogram, value['histograms']) for key, value in hist.items()}
            top_K.append([k for k, v in sorted(similarities_inter.items(), key=lambda item: item[1], reverse=True)][:K])
            
        else:
            similarities_inter = {key: measures.bhattacharyyaDistance(histogram, value['histograms']) for key, value in hist.items()}
            top_K.append([k for k, v in sorted(similarities_inter.items(), key=lambda item: item[1], reverse=False)][:K])
    
    top_K_num = [process_name(name) for name in top_K] 

    mapk_K = mapk(labels, top_K_num, K)


    return mapk_K


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

    with open('data/qsd1_w1/gt_corresps.pkl', 'rb') as f:
        labels = pickle.load(f)

    # Calculate similarities with CIELAB histograms
    # First, using histogram Intersection

    print("Calculating mAP@1 using CIELAB and histogram intersection...")
    mapInterCIELAB_1 = mAPK(1, histograms_cielab, labels, "intersection", hist_type="CIELAB")
    print("mAP@1 for CIELAB and Intersection: ", mapInterCIELAB_1)


    print("Calculating mAP@5 using CIELAB and histogram intersection...")
    mapInterCIELAB_5 = mAPK(5, histograms_cielab, labels, "intersection", hist_type="CIELAB")
    print("mAP@5 for CIELAB and Intersection: ", mapInterCIELAB_5)

    # Second, using Bhattacharyya distance

    print("Calculating mAP@1 using CIELAB and Bhattacharyya distance...")
    mapBhattCIELAB_1 = mAPK(1, histograms_cielab, labels, "bhatt", hist_type="CIELAB")
    print("mAP@1 for CIELAB and Bhatt", mapBhattCIELAB_1)

    print("Calculating mAP@5 using CIELAB and Bhattacharyya distance...")
    mapBhattCIELAB_5 = mAPK(5, histograms_cielab, labels, "bhatt", hist_type="CIELAB")
    print("mAP@5 for CIELAB and Bhatt", mapBhattCIELAB_5)

    # Calculate similarities with HSV histograms
    # First, using histogram Intersection

    print("Calculating mAP@1 using HSV and histogram intersection...")
    mapInterHSV_1 = mAPK(1, histograms_hsv, labels, "intersection", hist_type="HSV")
    print("mAP@1 for HSV and Intersection: ", mapInterHSV_1)


    print("Calculating mAP@5 using HSV and histogram intersection...")
    mapInterHSV_5 = mAPK(5, histograms_hsv, labels, "intersection", hist_type="HSV")
    print("mAP@5 for HSV and Bhatt: ", mapInterHSV_5)

    # Second, using Bhattacharyya distance

    print("Calculating mAP@1 using HSV and Bhattacharyya distance...")
    mapBhattHSV_1 = mAPK(1, histograms_hsv, labels, "bhatt", hist_type="HSV")
    print("mAP@1 for HSV and Bhatt: ", mapBhattHSV_1)


    print("Calculating mAP@5 using HSV and Bhattacharyya distance...")
    mapBhattHSV_5 = mAPK(5, histograms_hsv, labels, "bhatt", hist_type="HSV")
    print("mAP@5 for HSV and Bhatt: ", mapBhattHSV_5)
