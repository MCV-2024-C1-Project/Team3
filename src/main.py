import os
import sys
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
RESULTS_FOLDER = './results'
GT_CORRESPS_FILE = os.path.join(QSD1_W1_FOLDER, 'gt_corresps.pkl')

HLS_HIST_NPY = os.path.join(RESULTS_FOLDER, 'histograms_hls.npy')
HSV_HIST_NPY = os.path.join(RESULTS_FOLDER, 'histograms_hsv.npy')


def load_histograms(histogram_path, descriptor, folder_path):
    """Load histograms from file if available, otherwise calculate them."""
    if os.path.exists(histogram_path):
        print(f"Loading histograms from {histogram_path}...")
        histograms = np.load(histogram_path, allow_pickle=True).item()
    else:
        print(f"Calculating histograms for {descriptor.color_space}...")
        histograms = process_images(folder_path, descriptor)
        np.save(histogram_path, histograms)
        print(f"Histograms saved to {histogram_path}")
    return histograms


def process_images(folder_path, descriptor):
    """Calculate histograms for all images in the folder."""
    histograms_dict = {}
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(".jpg"):
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error loading image {image_path}")
                continue
            hist = descriptor.describe(image)
            histograms_dict[filename] = {
                'color_space': descriptor.color_space,
                'histogram_bins': descriptor.histogram_bins,
                'histograms': hist
            }
    return histograms_dict


def calculate_similarity(histograms, descriptor, labels, K, similarity_measure):
    """Calculate mAP@K for a given similarity measure and descriptor."""
    measures = ComputeSimilarity()
    top_K = []

    for img in sorted(os.listdir(QSD1_W1_FOLDER)):
        image_path = os.path.join(QSD1_W1_FOLDER, img)

        # Check if the file is an image
        if not img.endswith(".jpg"):
            continue
        
        try:
            histogram = descriptor.describe(cv2.imread(image_path))
        except Exception as e:
            print(f"Error processing image {img}: {e}")
            continue

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

    return mapk(labels, top_K, K)



def process_similarity_measures(histograms_hls, histograms_hsv, labels):
    """Process all combinations of descriptors and similarity measures."""
    descriptors = {
        "HLS": ImageDescriptor('HLS'),
        "HSV": ImageDescriptor('HSV')
    }

    similarity_measures = ["intersection", "bhatt", "canberra"]

    for hist_type, histograms in zip(["HLS", "HSV"], [histograms_hls, histograms_hsv]):
        descriptor = descriptors[hist_type]
        for measure in similarity_measures:
            print(f"Calculating mAP@1 for {hist_type} and {measure}...")
            mAP_1 = calculate_similarity(histograms, descriptor, labels, 1, measure)
            print(f"mAP@1 for {hist_type} and {measure}: {mAP_1}")

            print(f"Calculating mAP@5 for {hist_type} and {measure}...")
            mAP_5 = calculate_similarity(histograms, descriptor, labels, 5, measure)
            print(f"mAP@5 for {hist_type} and {measure}: {mAP_5}")


if __name__ == '__main__':
    # Create results folder if it doesn't exist
    if not os.path.exists(RESULTS_FOLDER):
        os.makedirs(RESULTS_FOLDER)

    # Load histograms for HLS and HSV
    histograms_hls = load_histograms(HLS_HIST_NPY, ImageDescriptor('HLS'), BBDD_FOLDER)
    histograms_hsv = load_histograms(HSV_HIST_NPY, ImageDescriptor('HSV'), BBDD_FOLDER)

    # Load the ground truth labels
    with open(GT_CORRESPS_FILE, 'rb') as f:
        labels = pickle.load(f)

    # Process all combinations of descriptors and similarity measures
    process_similarity_measures(histograms_hls, histograms_hsv, labels)
