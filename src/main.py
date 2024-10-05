import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import cv2
import numpy as np
from descriptors import ImageDescriptor
import pickle
from similarity import ComputeSimilarity

from evaluation.average_precision import mapk

# Constants for the paths to the data (using path.join to avoid problems with the OS :D )
DATA_FOLDER = './data'
BBDD_FOLDER = os.path.join(DATA_FOLDER, 'BBDD')
QSD1_W1_FOLDER = os.path.join(DATA_FOLDER, 'qsd1_w1')
RESULTS_FOLDER = './results'
GT_CORRESPS_FILE = os.path.join(QSD1_W1_FOLDER, 'gt_corresps.pkl')
CIELAB_HIST_NPY = os.path.join(RESULTS_FOLDER, 'histograms_cielab.npy')
HSV_HIST_NPY = os.path.join(RESULTS_FOLDER, 'histograms_hsv.npy')


def process_name(name): 
    numbers = []
    for n in name:
        number_str = n[5:-4]  # "00060"
        number = int(number_str)  # Convertir a entero para eliminar los ceros a la izquierda
        numbers.append(number)
    return numbers


def process_images(folder_path, descriptor):
    histograms_dict = {}
    
    # Process each image in the folder
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(".jpg"):  # Filter only jpg files
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)
            
            if image is None:
                print(f"Error loading image {image_path}")
                continue
            
            # Get the descriptor histogram
            hist = descriptor.describe(image)
            
            # Save the histogram as an image
            output_image_path = os.path.join(RESULTS_FOLDER, f'{filename.split(".")[0]}_{descriptor.color_space.lower()}_hist.png')
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
    measures = ComputeSimilarity()

    for img in sorted(os.listdir(QSD1_W1_FOLDER)):
        image_path = os.path.join(QSD1_W1_FOLDER, img)
        
        if hist_type == "CIELAB":
            descriptor = ImageDescriptor('CIELAB')
        else:
            descriptor = ImageDescriptor('HSV')

        try:
            histogram = descriptor.describe(cv2.imread(image_path))
        except:
            continue
        
        if similarity_measure == "intersection":  
            similarities = {key: measures.histogramIntersection(histogram, value['histograms']) for key, value in hist.items()}
            top_K.append([k for k, v in sorted(similarities.items(), key=lambda item: item[1], reverse=True)][:K])
        elif similarity_measure == "canberra":
            similarities = {key: measures.canberraDistance(histogram, value['histograms']) for key, value in hist.items()}
            top_K.append([k for k, v in sorted(similarities.items(), key=lambda item: item[1], reverse=False)][:K])
        else:
            similarities = {key: measures.bhattacharyyaDistance(histogram, value['histograms']) for key, value in hist.items()}
            top_K.append([k for k, v in sorted(similarities.items(), key=lambda item: item[1], reverse=False)][:K])
    
    top_K_num = [process_name(name) for name in top_K] 
    mapk_K = mapk(labels, top_K_num, K)

    return mapk_K


if __name__ == '__main__':
    # Create results folder if it doesn't exist
    if not os.path.exists(RESULTS_FOLDER):
        os.makedirs(RESULTS_FOLDER)

    # Check if histograms have already been calculated
    if os.path.exists(CIELAB_HIST_NPY):
        print(f"Loading CIELAB histograms from {CIELAB_HIST_NPY}...")
        histograms_cielab = np.load(CIELAB_HIST_NPY, allow_pickle=True).item()
    else:
        print("Calculating CIELAB histograms...")
        cielab_descriptor = ImageDescriptor('CIELAB')
        histograms_cielab = process_images(BBDD_FOLDER, cielab_descriptor)
        np.save(CIELAB_HIST_NPY, histograms_cielab)
        print(f"CIELAB histograms saved as {CIELAB_HIST_NPY}")

    if os.path.exists(HSV_HIST_NPY):
        print(f"Loading HSV histograms from {HSV_HIST_NPY}...")
        histograms_hsv = np.load(HSV_HIST_NPY, allow_pickle=True).item()
    else:
        print("Calculating HSV histograms...")
        hsv_descriptor = ImageDescriptor('HSV')
        histograms_hsv = process_images(BBDD_FOLDER, hsv_descriptor)
        np.save(HSV_HIST_NPY, histograms_hsv)
        print(f"HSV histograms saved as {HSV_HIST_NPY}")

    print("All histograms have been successfully loaded or calculated.")

    # Load the labels for the comparison
    with open(GT_CORRESPS_FILE, 'rb') as f:
        labels = pickle.load(f)

    # Calculate similarities with CIELAB histograms
    print("Calculating mAP@1 using CIELAB and histogram intersection...")
    mapInterCIELAB_1 = mAPK(1, histograms_cielab, labels, "intersection", hist_type="CIELAB")
    print("mAP@1 for CIELAB and Intersection: ", mapInterCIELAB_1)

    print("Calculating mAP@5 using CIELAB and histogram intersection...")
    mapInterCIELAB_5 = mAPK(5, histograms_cielab, labels, "intersection", hist_type="CIELAB")
    print("mAP@5 for CIELAB and Intersection: ", mapInterCIELAB_5)

    print("Calculating mAP@1 using CIELAB and Bhattacharyya distance...")
    mapBhattCIELAB_1 = mAPK(1, histograms_cielab, labels, "bhatt", hist_type="CIELAB")
    print("mAP@1 for CIELAB and Bhatt", mapBhattCIELAB_1)

    print("Calculating mAP@5 using CIELAB and Bhattacharyya distance...")
    mapBhattCIELAB_5 = mAPK(5, histograms_cielab, labels, "bhatt", hist_type="CIELAB")
    print("mAP@5 for CIELAB and Bhatt", mapBhattCIELAB_5)

    print("Calculating mAP@1 using CIELAB and Canberra distance...")
    mapBhattCIELAB_1 = mAPK(1, histograms_cielab, labels, "canberra", hist_type="CIELAB")
    print("mAP@1 for CIELAB and Canberra", mapBhattCIELAB_1)

    print("Calculating mAP@5 using CIELAB and Canberra distance...")
    mapBhattCIELAB_5 = mAPK(5, histograms_cielab, labels, "canberra", hist_type="CIELAB")
    print("mAP@5 for CIELAB and Canberra", mapBhattCIELAB_5)

    # Calculate similarities with HSV histograms
    print("Calculating mAP@1 using HSV and histogram intersection...")
    mapInterHSV_1 = mAPK(1, histograms_hsv, labels, "intersection", hist_type="HSV")
    print("mAP@1 for HSV and Intersection: ", mapInterHSV_1)

    print("Calculating mAP@5 using HSV and histogram intersection...")
    mapInterHSV_5 = mAPK(5, histograms_hsv, labels, "intersection", hist_type="HSV")
    print("mAP@5 for HSV and Intersection: ", mapInterHSV_5)

    print("Calculating mAP@1 using HSV and Bhattacharyya distance...")
    mapBhattHSV_1 = mAPK(1, histograms_hsv, labels, "bhatt", hist_type="HSV")
    print("mAP@1 for HSV and Bhatt: ", mapBhattHSV_1)

    print("Calculating mAP@5 using HSV and Bhattacharyya distance...")
    mapBhattHSV_5 = mAPK(5, histograms_hsv, labels, "bhatt", hist_type="HSV")
    print("mAP@5 for HSV and Bhatt: ", mapBhattHSV_5)

    print("Calculating mAP@1 using HSV and Canberra distance...")
    mapBhattHSV_1 = mAPK(1, histograms_hsv, labels, "canberra", hist_type="HSV")
    print("mAP@1 for HSV and Canberra: ", mapBhattHSV_1)

    print("Calculating mAP@5 using HSV and Canberra distance...")
    mapBhattHSV_5 = mAPK(5, histograms_hsv, labels, "canberra", hist_type="HSV")
    print("mAP@5 for HSV and Canberra: ", mapBhattHSV_5)
