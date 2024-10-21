import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import cv2
import numpy as np
import pickle
import argparse
from descriptors import ImageDescriptor
from similarity import ComputeSimilarity
from evaluation.average_precision import mapk
from background_removal import CalculateBackground


# Argument parser setup
parser = argparse.ArgumentParser(description='Process image folder for similarity.')
parser.add_argument('images_folder', type=str, help='Path to the image folder (e.g., ./data/qsd1_w1)')
parser.add_argument('dimension', type=str, help='Histogram dimension (e.g., 3D')
parser.add_argument('colorspace', type=str, help='Histogram colorspace (e.g., HSV')
parser.add_argument('structure', type=str, help='Histogram structure (e.g., block')
parser.add_argument('measure', type=str, help='Similarity measure (e.g., intersection')
args = parser.parse_args()

# Define input folder based on argument
qsd_folder = args.images_folder
dimension = args.dimension
colorspace = args.colorspace
structure = args.structure
measure = args.measure
    

# Constants for paths
DATA_FOLDER = './data'
BBDD_FOLDER = os.path.join(DATA_FOLDER, 'BBDD')

QST1_W2_FOLDER = os.path.join(DATA_FOLDER, 'qst1_w2')
QST2_W2_FOLDER = os.path.join(DATA_FOLDER, 'qst2_w1')
RESULTS_FOLDER = './results'
GT_CORRESPS_FILE = os.path.join(qsd_folder, 'gt_corresps.pkl')

MASK_FOLDER = './masks'
NO_BG_FOLDER = './data/qsd2_w3_no_bg'

METHOD1_FOLDER = os.path.join(RESULTS_FOLDER, 'method1')  # Output folder for method1
METHOD2_FOLDER = os.path.join(RESULTS_FOLDER, 'method2')  # Output folder for method2


def load_histograms(dimension, structure, descriptor, folder_path):
    """Load histograms from file if available, otherwise calculate them."""
    histogram_path = '_'.join(('results/histograms', dimension, descriptor.color_space, structure)) + '.npy'
    if os.path.exists(histogram_path):
        print(f"Loading histograms from {histogram_path}...")
        histograms = np.load(histogram_path, allow_pickle=True).item()
    else:
        print(f"Calculating histograms for {dimension}, {structure}, {descriptor.color_space}...")
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
            hist = descriptor.describe(image, dimension, structure)
            histograms_dict[filename] = {
                'color_space': descriptor.color_space,
                'dimension': dimension,
                'structure': structure,
                'histogram_bins': descriptor.histogram_bins,
                'histograms': hist
            }
    return histograms_dict

def calculate_similarity(histograms, descriptor, labels, K, similarity_measure, dimension, structure, level, mask, folder=qsd_folder):
    """Calculate mAP@K for a given similarity measure and descriptor."""
    measures = ComputeSimilarity()
    top_K = []

    for img in sorted(os.listdir(folder)):
        image_path = os.path.join(folder, img)

        # Check if the file is an image
        if not img.endswith(".jpg") and not img.endswith(".png"):
            continue
        
        try:
            histogram = descriptor.describe(cv2.imread(image_path), dimension, structure)
        except Exception as e:
            print(f"Error processing image {img}: {e}")
            continue
        
        if structure == 'simple':
            if similarity_measure == "intersection":
                similarities = {key: measures.histogramIntersection(histogram, value['histograms']) for key, value in histograms.items()}
                reverse = True
            elif similarity_measure == "bhattacharyya":
                similarities = {key: measures.bhattacharyyaDistance(histogram, value['histograms']) for key, value in histograms.items()}
                reverse = True
            elif similarity_measure == "Chisqr":
                similarities = {key: measures.histogramChisqr(histogram, value['histograms']) for key, value in histograms.items()}
                reverse = True
            elif similarity_measure == "Correl":
                similarities = {key: measures.histogramCorrel(histogram, value['histograms']) for key, value in histograms.items()}
                reverse = False

            top_k = [k for k, v in sorted(similarities.items(), key=lambda item: item[1], reverse=reverse)][:K]
            top_k_numbers = [int(filename[5:-4]) for filename in top_k]
            top_K.append(top_k_numbers)

        elif structure == 'block' or structure == 'heriarchical':
            if similarity_measure == "intersection":
                similarities = {key: measures.histogramIntersection(np.array(histogram[level]['histogram'], dtype=np.float32).flatten(), 
                                                                     np.array(value['histograms'][level]['histogram'], dtype=np.float32).flatten()) 
                                for key, value in histograms.items()}
                reverse = True
            elif similarity_measure == "bhattacharyya":
                similarities = {key: measures.bhattacharyyaDistance(np.array(histogram[level]['histogram'], dtype=np.float32).flatten(), 
                                                                     np.array(value['histograms'][level]['histogram'], dtype=np.float32).flatten()) 
                                for key, value in histograms.items()}
                reverse = False
            elif similarity_measure == "Chisqr":
                similarities = {key: measures.histogramChisqr(np.array(histogram[level]['histogram'], dtype=np.float32).flatten(), 
                                                                     np.array(value['histograms'][level]['histogram'], dtype=np.float32).flatten()) 
                                for key, value in histograms.items()}
                reverse = False

            elif similarity_measure == "Correl":
                similarities = {key: measures.histogramCorrel(np.array(histogram[level]['histogram'], dtype=np.float32).flatten(), 
                                                                np.array(value['histograms'][level]['histogram'], dtype=np.float32).flatten()) 
                                for key, value in histograms.items()}
                reverse = True

            top_k = [k for k, v in sorted(similarities.items(), key=lambda item: item[1], reverse=reverse)][:K]
            top_k_numbers = [int(filename[5:-4]) for filename in top_k]
            top_K.append(top_k_numbers)

    return top_K  # Return top K results (list of lists)


def process_similarity_measures(histograms, descriptor, labels, dimension, structure, k_val, method_folder, measure, mask, images_folder=qsd_folder):
    """Process all combinations of similarity measures for a single descriptor."""
    # similarity_measures = ["intersection", "canberra"]

    # for measure in similarity_measures:
    if structure == 'simple':
        top_K = calculate_similarity(histograms, descriptor, labels, k_val, measure, dimension, structure, None, mask=mask, folder=images_folder)
        map_k = mapk(labels, top_K, k_val)
        print(f"mAP@{k_val} for {dimension}, {structure}, {descriptor.color_space} and {measure}: {map_k}")
    
    elif structure == 'block' or structure == 'heriarchical':
        for level in range(3):
            top_K = calculate_similarity(histograms, descriptor, labels, k_val, measure, dimension, structure, level,mask=mask, folder=images_folder)
            map_k = mapk(labels, top_K, k_val)
            # Calculate mAP
            print(f"mAP@{k_val} for {dimension}, {structure}, {descriptor.color_space} at level {level} and {measure}: {map_k}")

    if not os.path.exists(method_folder):
        os.makedirs(method_folder)

    pkl_output_path = os.path.join(method_folder, 'result.pkl')
    with open(pkl_output_path, 'wb') as f:
        pickle.dump(top_K, f)

    print(f"Results saved to {pkl_output_path}")
     
def compute_confusion_matrix(ground_truth, predicted):
    # Ensure ground_truth and predicted are NumPy arrays of the same shape
    ground_truth = np.asarray(ground_truth)
    predicted = np.asarray(predicted)
    
    if ground_truth.shape != predicted.shape:
        raise ValueError("The ground truth and predicted masks must have the same shape.")
    
    # Compute the confusion matrix components
    TP = np.sum((ground_truth == 255) & (predicted == 255))
    TN = np.sum((ground_truth == 0) & (predicted == 0))
    FP = np.sum((ground_truth == 0) & (predicted == 255))
    FN = np.sum((ground_truth == 255) & (predicted == 0))
    
    return TP, TN, FP, FN

def compute_precision_recall_f1(ground_truth, predicted):

    TP, TN, FP, FN = compute_confusion_matrix(ground_truth, predicted)
    # Calculate Precision
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    
    # Calculate Recall
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    
    # Calculate F1 Score
    if precision + recall > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0
    
    return precision, recall, f1_score


def background_images(qsd_folder):
    # Calculate background images
    final_image=None
    if qsd_folder == "./data/qsd2_w3" or qsd_folder == "./data\qst2_w1" or qsd_folder == "./data/qst2_w1":
        print("Removing background")
        NO_BG_FOLDER = './data/qsd2_w3_no_bg'
        if not os.path.exists(NO_BG_FOLDER):
            os.makedirs(NO_BG_FOLDER)
        
        # I/O and metrics initialization
        iou_scores = []
        precisions = []
        recalls = []
        f1s = []

        for image_name in os.listdir(qsd_folder):
            if image_name.endswith(".jpg"):
                image = cv2.imread(os.path.join(qsd_folder, image_name))
                if image is None:
                    print(f"Error loading image {image_name}")
                    continue
                
                background = CalculateBackground(image)
                seed_points = [(0, 0), (image.shape[1] - 1, 0), (0, image.shape[0] - 1), (image.shape[1] - 1, image.shape[0] - 1)]

                edge_map = background.adaptive_thresholding(image)
                tot_mask = np.zeros(image.shape[:2], dtype=np.uint8)

                for seed in seed_points:
                    mask = background.flood_fill_region_with_edges(seed, tolerance=5, edge_map=edge_map)
                    tot_mask = np.maximum(tot_mask, mask)

                foreground = background.apply_mask(tot_mask)
                final_mask = background.color_thresholding_simple(0, foreground)
                tot_mask = tot_mask + final_mask
                final_image = background.morphological_operations_cleanse(tot_mask)
                final_image = cv2.bitwise_not(final_image)

                cv2.imwrite(os.path.join(MASK_FOLDER, image_name.split('.')[0] + ".png"), final_image)

                if qsd_folder == "./data/qsd2_w3":
                    gt = cv2.imread(os.path.join("./data/qsd2_w3/", image_name[:-4] + ".png"), cv2.IMREAD_GRAYSCALE)
                    if gt is not None:
                        intersection = np.logical_and(gt, final_image)
                        union = np.logical_or(gt, final_image)
                        iou_score = np.sum(intersection) / np.sum(union) if np.sum(union) > 0 else 0
                        iou_scores.append(iou_score)

                        precision, recall, f1 = compute_precision_recall_f1(gt, final_image)
                        precisions.append(precision)
                        recalls.append(recall)
                        f1s.append(f1)

                # image_bgra = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
                # image_bgra[final_image == 0, 3] = 0
                # image_bgra[final_image != 0, 3] = 255
                masked_image = cv2.bitwise_and(image, image, mask=final_image)
                x, y, w, h = cv2.boundingRect(final_image)
                cropped_image = masked_image[y:y+h, x:x+w]
                output_path = os.path.join(NO_BG_FOLDER, image_name.split('.')[0] + ".png")
                
                cv2.imwrite(output_path, cropped_image)

        qsd_folder = NO_BG_FOLDER
        # Safely compute mean metrics
        if iou_scores:
            print(f"Mean IoU: {np.mean(iou_scores)}")
        if precisions:
            print(f"Mean Precision: {np.mean(precisions)}")
        if recalls:
            print(f"Mean Recall: {np.mean(recalls)}")
        if f1s:
            print(f"Mean F1 Score: {np.mean(f1s)}")
            
    return qsd_folder, final_image


if __name__ == '__main__':
    if not os.path.exists(RESULTS_FOLDER):
        os.makedirs(RESULTS_FOLDER)
        
    
    # Load histograms for HLS and HSV
    # histograms_hls = load_histograms(HLS_HIST_NPY, ImageDescriptor('HLS'), BBDD_FOLDER)
    # histograms_hsv = load_histograms(HSV_HIST_NPY, ImageDescriptor('HSV'), BBDD_FOLDER)
    histograms = load_histograms(dimension, structure, ImageDescriptor(colorspace), BBDD_FOLDER)

    # Attempt to load the ground truth labels, if the file exists
    labels = None
    try:
        with open(GT_CORRESPS_FILE, 'rb') as f:
            labels = pickle.load(f)
        print("Ground truth labels loaded successfully from", GT_CORRESPS_FILE)
    except FileNotFoundError:
        print(f"Warning: Ground truth file {GT_CORRESPS_FILE} not found. Continuing without labels.")


    print(qsd_folder)
    qsd_folder, mask = background_images(qsd_folder)
    # After all masks have been applied and evaluated, process similarity with the background-removed images
    print("Processing similarity using method:", colorspace, "dimension:", dimension, "structure:", structure)
    print(qsd_folder)
    process_similarity_measures(histograms, ImageDescriptor(colorspace), labels, dimension, structure, k_val=1, mask=cv2.bitwise_not(mask), measure=measure, method_folder=METHOD1_FOLDER, images_folder=qsd_folder)
                


    # print("Processing similarity for test 1 using method:", colorspace, "dimension:", dimension, "structure:", structure)
    # # Process similarity measures using the HLS descriptor and the top-K similarity
    # process_similarity_measures(histograms, ImageDescriptor(colorspace), labels, dimension, structure, k_val=10, mask=None, measure=measure, method_folder=METHOD1_FOLDER, images_folder=QST1_W2_FOLDER)
    
    # qsd_folder = QST2_W2_FOLDER
    # print(qsd_folder)
    # qsd_folder, mask = background_images(qsd_folder)
    # print("Processing similarity for test 2 using method:", colorspace, "dimension:", dimension, "structure:", structure)
    # # Process similarity measures using the HLS descriptor and the top-K similarity
    # process_similarity_measures(histograms, ImageDescriptor(colorspace), labels, dimension, structure, k_val=10, mask=cv2.bitwise_not(mask), measure=measure, method_folder=METHOD2_FOLDER, images_folder=NO_BG_FOLDER)
