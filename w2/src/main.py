import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import cv2
import numpy as np
import pickle
from descriptors import ImageDescriptor
from similarity import ComputeSimilarity
from evaluation.average_precision import mapk
from background_removal import CalculateBackground

# Constants for paths
DATA_FOLDER = './data'
BBDD_FOLDER = os.path.join(DATA_FOLDER, 'BBDD')
QSD1_W1_FOLDER = os.path.join(DATA_FOLDER, 'qsd1_w1')
QST1_W1_FOLDER = os.path.join(DATA_FOLDER, 'qst1_w1')
RESULTS_FOLDER = './results'
GT_CORRESPS_FILE = os.path.join(QSD1_W1_FOLDER, 'gt_corresps.pkl')

MASK_FOLDER = './masks'

HLS_HIST_NPY = os.path.join(RESULTS_FOLDER, 'histograms_hls.npy')
HSV_HIST_NPY = os.path.join(RESULTS_FOLDER, 'histograms_hsv.npy')
METHOD1_FOLDER = os.path.join(RESULTS_FOLDER, 'method1')  # Output folder for HLS
METHOD2_FOLDER = os.path.join(RESULTS_FOLDER, 'method2')  # Output folder for HSV

# Print paths to verify


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

def calculate_similarity(histograms, descriptor, labels, K, similarity_measure, folder=QSD1_W1_FOLDER):
    """Calculate mAP@K for a given similarity measure and descriptor."""
    measures = ComputeSimilarity()
    top_K = []

    for img in sorted(os.listdir(folder)):
        image_path = os.path.join(folder, img)

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

    return top_K  # Return top K results (list of lists)

def process_similarity_measures(histograms, descriptor, labels, k_val, method_folder, images_folder=QSD1_W1_FOLDER):
    """Process all combinations of similarity measures for a single descriptor."""
    similarity_measures = ["intersection", "canberra"]

    for measure in similarity_measures:
        print(f"Calculating mAP@{k_val} for {descriptor.color_space} and {measure}...")
        top_K = calculate_similarity(histograms, descriptor, labels, k_val, measure, folder=images_folder)

        # Calculate mAP
        map_k = mapk(labels, top_K, k_val)
        print(f"mAP@{k_val} for {descriptor.color_space} and {measure}: {map_k}")
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

    # Load histograms for HLS and HSV
    histograms_hls = load_histograms(HLS_HIST_NPY, ImageDescriptor('HLS'), BBDD_FOLDER)
    histograms_hsv = load_histograms(HSV_HIST_NPY, ImageDescriptor('HSV'), BBDD_FOLDER)

    # Load the ground truth labels
    with open(GT_CORRESPS_FILE, 'rb') as f:
        labels = pickle.load(f)

    # Process results for HLS (method1) with k=1 and k=5
    print("Processing results for method1 (HLS)...")
    #process_similarity_measures(histograms_hls, ImageDescriptor('HLS'), labels, k_val=1, method_folder=METHOD1_FOLDER)
    #process_similarity_measures(histograms_hls, ImageDescriptor('HLS'), labels, k_val=5, method_folder=METHOD1_FOLDER)

    # Process results for HSV (method2) with k=1 and k=5
    print("Processing results for method2 (HSV)...")
    #process_similarity_measures(histograms_hsv, ImageDescriptor('HSV'), labels, k_val=1, method_folder=METHOD2_FOLDER)
    #process_similarity_measures(histograms_hsv, ImageDescriptor('HSV'), labels, k_val=5, method_folder=METHOD2_FOLDER)
    
    # Process results for HLS (method1) with k=10 for the test
    print("Processing results for method1 (HLS)...")
    #process_similarity_measures(histograms_hls, ImageDescriptor('HLS'), labels, k_val=10, method_folder=METHOD1_FOLDER, images_folder=QST1_W1_FOLDER)

    # Process results for HSV (method2) with k=1 and k=10 for the test
    print("Processing results for method2 (HSV)...")
    #process_similarity_measures(histograms_hsv, ImageDescriptor('HSV'), labels, k_val=10, method_folder=METHOD2_FOLDER, images_folder=QST1_W1_FOLDER)

    # Calculate background images

    iou_scores = []

    for image_name in os.listdir("./data/qsd2_w2"):

        if image_name.endswith(".jpg"):

            image = cv2.imread("./data/qsd2_w2/"+image_name)
            background = CalculateBackground(image)


            seed_points = [
                (0, 0),  # Top-left corner
                (image.shape[1] - 1, 0),  # Top-right corner
                (0, image.shape[0] - 1),  # Bottom-left corner
                (image.shape[1] - 1, image.shape[0] - 1),  # Bottom-right corner
            ]

            edge_map = background.adaptive_thresholding(image)

            #background.display_image(edge_map, "Adaptive Gaussian Thresholding mask")

        # Perform region growing for each seed point and stack the masks
            tot_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            for seed in seed_points:
                mask = background.flood_fill_region_with_edges(seed, tolerance=5, edge_map=edge_map)
                tot_mask = np.maximum(tot_mask, mask)

            # Apply the mask to get the foreground
            foreground = background.apply_mask(tot_mask)

            final_mask = background.color_thresholding_simple(0, foreground)

            tot_mask = tot_mask + final_mask

            final_image = background.morphological_operations_cleanse(tot_mask)
            final_image = cv2.bitwise_not(final_image)

            cv2.imwrite(MASK_FOLDER+"/"+image_name, final_image)

            # Load ground truth
            gt = cv2.imread("./data/qsd2_w2/"+image_name[:-4]+".png", cv2.IMREAD_GRAYSCALE)

            # Calculate IoU
            intersection = np.logical_and(gt, final_image)
            union = np.logical_or(gt, final_image)
            iou_score = np.sum(intersection) / np.sum(union)

            iou_scores.append(iou_score)

    print(f"Mean IoU: {np.mean(iou_scores)}")