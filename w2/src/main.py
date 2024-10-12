import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import cv2
import numpy as np
import pickle
from descriptors import ImageDescriptor
from similarity import ComputeSimilarity
from evaluation.average_precision import mapk

# Constants for paths
DATA_FOLDER = './data'
BBDD_FOLDER = os.path.join(DATA_FOLDER, 'BBDD')
QSD1_W1_FOLDER = os.path.join(DATA_FOLDER, 'qsd1_w1')
QST1_W1_FOLDER = os.path.join(DATA_FOLDER, 'qst1_w1')
RESULTS_FOLDER = './results'
GT_CORRESPS_FILE = os.path.join(QSD1_W1_FOLDER, 'gt_corresps.pkl')

# HLS_HIST_NPY = os.path.join(RESULTS_FOLDER, 'histograms_hls.npy')
# HSV_HIST_NPY = os.path.join(RESULTS_FOLDER, 'histograms_hsv.npy')
METHOD1_FOLDER = os.path.join(RESULTS_FOLDER, 'method1')  # Output folder for HLS
METHOD2_FOLDER = os.path.join(RESULTS_FOLDER, 'method2')  # Output folder for HSV

def load_histograms(dimension,structure, descriptor, folder_path):
    """Load histograms from file if available, otherwise calculate them."""
    histogram_path='_'.join(('results/histograms',dimension,descriptor.color_space,structure))+'.npy' #ex: histograms_3D_HSV_heriarchical.npy
    if os.path.exists(histogram_path):
        print(f"Loading histograms from {histogram_path}...")
        histograms = np.load(histogram_path, allow_pickle=True).item()
    else:
        print(f"Calculating histograms for {dimension,structure,descriptor.color_space}...")
        histograms = process_images(folder_path, dimension, structure, descriptor)
        np.save(histogram_path, histograms)
        print(f"Histograms saved to {histogram_path}")
    return histograms

def process_images(folder_path, dimension, structure, descriptor):
    """Calculate histograms for all images in the folder."""
    histograms_dict = {}
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(".jpg"):

            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error loading image {image_path}")
                continue
            hist = descriptor.describe(image,dimension,structure)
            
            histograms_dict[filename] = {
                'color_space': descriptor.color_space,
                'dimension': dimension,
                'structure': structure,
                'histogram_bins': descriptor.histogram_bins,
                'histograms': hist
            }
    return histograms_dict

def calculate_similarity(histograms, descriptor, labels, K, similarity_measure,dimension,structure,level, folder=QSD1_W1_FOLDER):
    """Calculate mAP@K for a given similarity measure and descriptor."""
    measures = ComputeSimilarity()
    top_K = []

    for img in sorted(os.listdir(folder)):
        image_path = os.path.join(folder, img)

        # Check if the file is an image
        if not img.endswith(".jpg"):
            continue
        
        try:
            histogram = descriptor.describe(cv2.imread(image_path),dimension,structure)
        except Exception as e:
            print(f"Error processing image {img}: {e}")
            continue
        
        if structure=='simple':
            if similarity_measure == "intersection":
                similarities = {key: measures.histogramIntersection(histogram, value['histograms']) for key, value in histograms.items()}
                reverse = True
            elif similarity_measure == "bhatt":
                similarities = {key: measures.bhattacharyyaDistance(histogram, value['histograms']) for key, value in histograms.items()}
                reverse = False
            elif similarity_measure == "canberra":
                similarities = {key: measures.canberraDistance(histogram, value['histograms']) for key, value in histograms.items()}
                reverse = False

            top_k = [k for k, v in sorted(similarities.items(), key=lambda item: item[1], reverse=reverse)][:K]
            top_k_numbers = [int(filename[5:-4]) for filename in top_k]
            top_K.append(top_k_numbers)

        elif structure=='block' or structure=='heriarchical':
            if similarity_measure == "intersection":                
                similarities = {key: measures.histogramIntersection(np.array(histogram[level]['histogram'],dtype=np.float32).flatten(), np.array(value['histograms'][level]['histogram'],dtype=np.float32).flatten()) for key, value in histograms.items()}
                reverse = True
            elif similarity_measure == "bhatt":
                similarities = {key: measures.bhattacharyyaDistance(histogram[level]['histogram'], value['histograms'][level]['histogram']) for key, value in histograms.items()}
                reverse = False
            elif similarity_measure == "canberra":
                similarities = {key: measures.canberraDistance(np.array(histogram[level]['histogram'],dtype=np.float32).flatten(), np.array(value['histograms'][level]['histogram'],dtype=np.float32).flatten()) for key, value in histograms.items()}
                reverse = False

            top_k = [k for k, v in sorted(similarities.items(), key=lambda item: item[1], reverse=reverse)][:K]
            top_k_numbers = [int(filename[5:-4]) for filename in top_k]
            top_K.append(top_k_numbers)

    return top_K  # Return top K results (list of lists)

def process_similarity_measures(histograms, descriptor, labels,dimension,structure, k_val, method_folder, images_folder=QSD1_W1_FOLDER):
    """Process all combinations of similarity measures for a single descriptor."""
    similarity_measures = ["intersection", "canberra"]

    for measure in similarity_measures:
        if structure=='simple':
            # print(f"Calculating mAP@{k_val} for {descriptor.color_space} and {measure}...")
            top_K = calculate_similarity(histograms, descriptor, labels, k_val, measure,dimension,structure,None, folder=images_folder)

            # Calculate mAP
            map_k = mapk(labels, top_K, k_val)
            print(f"mAP@{k_val} for {dimension, structure, descriptor.color_space} and {measure}: {map_k}")
            # print(f"List of lists for {descriptor.color_space} and {measure} (Top {k_val}): {top_K}\n")
        
        elif structure=='block' or structure=='heriarchical':
            for level in range(3):
                # print(f"Calculating mAP@{k_val} for {descriptor.color_space} and {measure}...")
                top_K = calculate_similarity(histograms, descriptor, labels, k_val, measure,dimension,structure,level, folder=images_folder,)

                # Calculate mAP
                map_k = mapk(labels, top_K, k_val)
                print(f"mAP@{k_val} for {dimension, structure, descriptor.color_space} at level {level} and {measure}: {map_k}")
                # print(f"List of lists for {descriptor.color_space} and {measure} (Top {k_val}): {top_K}\n")
        
        elif structure=='heriarchical':
            for level in range(3):
                # print(f"Calculating mAP@{k_val} for {descriptor.color_space} and {measure}...")
                top_K = calculate_similarity(histograms, descriptor, labels, k_val, measure,dimension,structure,level, folder=images_folder,)

                # Calculate mAP
                map_k = mapk(labels, top_K, k_val)
                print(f"mAP@{k_val} for {dimension, structure, descriptor.color_space} at level {level} and {measure}: {map_k}")
                # print(f"List of lists for {descriptor.color_space} and {measure} (Top {k_val}): {top_K}\n")


    # Save results to the corresponding method folder
    if not os.path.exists(method_folder):
        os.makedirs(method_folder)

    pkl_output_path = os.path.join(method_folder, 'result.pkl')
    with open(pkl_output_path, 'wb') as f:
        pickle.dump(top_K, f)

    print(f"Results saved to {pkl_output_path}")

if __name__ == '__main__':
    # Create results folder if it doesn't exist
    if not os.path.exists(RESULTS_FOLDER):
        os.makedirs(RESULTS_FOLDER)
    
    # Load the ground truth labels
    with open(GT_CORRESPS_FILE, 'rb') as f:
        labels = pickle.load(f)
    
    # Load histograms for HLS and HSV
    for dimension in ['3D','2D']:
        for colorspace in ['HSV','HLS']:
            for structure in ['block','heriarchical']:
                histograms = load_histograms(dimension,structure, ImageDescriptor(colorspace), BBDD_FOLDER)

    

                # Process results for HLS (method1) with k=1 and k=5
                print("Processing results for method1 for {} {} {}".format(colorspace, dimension,structure))
                process_similarity_measures(histograms, ImageDescriptor(colorspace), labels, dimension,structure, k_val=1, method_folder=METHOD1_FOLDER)
                process_similarity_measures(histograms, ImageDescriptor(colorspace), labels,dimension,structure, k_val=5, method_folder=METHOD1_FOLDER)

                
                # # Process results for HLS (method1) with k=10 for the test
                # print("Processing results for method1 (HLS)...")
                # process_similarity_measures(histograms_hls, ImageDescriptor('HLS'), labels, k_val=10, method_folder=METHOD1_FOLDER, images_folder=QST1_W1_FOLDER)

                # # Process results for HSV (method2) with k=1 and k=10 for the test
                # print("Processing results for method2 (HSV)...")
                # process_similarity_measures(histograms_hsv, ImageDescriptor('HSV'), labels, k_val=10, method_folder=METHOD2_FOLDER, images_folder=QST1_W1_FOLDER)
