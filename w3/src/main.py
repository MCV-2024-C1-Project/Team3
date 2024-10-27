import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import numpy as np
import pickle
import argparse
from descriptors import TextureDescriptor
from similarity import ComputeSimilarity
from evaluation.average_precision import mapk
from background_removal import CalculateBackground
from denoiser import denoiseAll, LinearDenoiser


# Argument parser setup
parser = argparse.ArgumentParser(description='Process image folder for similarity.')
parser.add_argument('images_folder', type=str, help='Path to the image folder (e.g., ./data/qsd1_w1)')
parser.add_argument('structure', type=str, help='Structure of the descriptor (e.g., block, DCT, LBP, heriarchical)')
parser.add_argument('colorspace', type=str, help='Histogram colorspace (e.g., HSV')
parser.add_argument('measure', type=str, help='Similarity measure (e.g., intersection')
args = parser.parse_args()

# Define input folder based on argument
qsd_folder = args.images_folder
structure = args.structure
colorspace = args.colorspace
quantization = False
measure = args.measure
    
# Constants for paths
DATA_FOLDER = './data'
RESULTS_FOLDER = './results'

BBDD_FOLDER = os.path.join(DATA_FOLDER, 'BBDD')

QSD1_W3_FOLDER = os.path.join(DATA_FOLDER, 'qsd1_w3')
QSD2_W3_FOLDER = os.path.join(DATA_FOLDER, 'qsd2_w3')
NO_BG_FOLDER =  os.path.join(DATA_FOLDER, 'qsd2_w3_no_bg')

DENOISED_IMAGES_1 = os.path.join(DATA_FOLDER, 'denoised_images_1')
DENOISED_IMAGES_2 = os.path.join(DATA_FOLDER, 'denoised_images_2')
MASK_FOLDER =  os.path.join(DATA_FOLDER, 'masks')

GT_CORRESPS_FILE = os.path.join(DATA_FOLDER, qsd_folder, 'gt_corresps.pkl')

METHOD1_FOLDER = os.path.join(RESULTS_FOLDER, 'method1')  # Output folder for method1
METHOD2_FOLDER = os.path.join(RESULTS_FOLDER, 'method2')  # Output folder for method2


def load_histograms(structure, descriptor, folder_path,quantization):
    """Load histograms from file if available, otherwise calculate them."""
    histogram_path = '_'.join(('results/descriptors', descriptor.color_space, structure,str(quantization))) + '.npy'
    if os.path.exists(histogram_path):
        print(f"Loading histograms from {histogram_path}...")
        histograms = np.load(histogram_path, allow_pickle=True).item()
    else:
        print(f"Calculating histograms for {structure}, {descriptor.color_space}, quantization {quantization}...")
        histograms = process_images(folder_path, structure, descriptor,quantization)
        np.save(histogram_path, histograms)
        print(f"Histograms saved to {histogram_path}")
    return histograms


def process_images(folder_path, structure, descriptor,quantization):
    """Calculate histograms for all images in the folder."""
    histograms_dict = {}
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(".jpg"):
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error loading image {image_path}")
                continue
            hist = descriptor.describe(image, structure,quantization)
            histograms_dict[filename] = {
                'color_space': descriptor.color_space,
                'quantization': quantization,
                'structure': structure,
                'descriptor': hist
            }
    return histograms_dict


def kullback_leibler_divergence(p, q):
    p = np.asarray(p)
    q = np.asarray(q)
    filt = np.logical_and(p != 0, q != 0)
    return np.sum(p[filt] * np.log2(p[filt] / q[filt]))

import random as rng

def detect_pictures(image, mask):
    """Detect possible cuadros (regions of interest) in the image using contours."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rng.seed(12345)


    cuadros = []
    i=0
    for contour in contours: # Filter out small areas (noise)
        # x, y, w, h = cv2.boundingRect(contour)
        # cuadro = image[y:y+h, x:x+w]

        epsilon = 0.1*cv2.arcLength(contour,True)
        pp = cv2.approxPolyDP(contour,epsilon,True)  
        p=np.array([[pp[0][0][0],pp[0][0][1]],[pp[1][0][0],pp[1][0][1]],[pp[2][0][0],pp[2][0][1]],[pp[3][0][0],pp[3][0][1]]])
        # Sort the points by y-coordinate (top to bottom)
        points = p[np.argsort(p[:, 1])]
        
        # Separate the points into top and bottom halves
        top_points = points[:2]
        bottom_points = points[2:]
        
        # Sort the top points by x-coordinate (left to right) to get upper-left and upper-right
        top_left, top_right = top_points[np.argsort(top_points[:, 0])]
        
        # Sort the bottom points by x-coordinate (left to right) to get lower-left and lower-right
        bottom_left, bottom_right = bottom_points[np.argsort(bottom_points[:, 0])]
        
        # Arrange the points in the desired order
        ordered_points = np.array([top_left, bottom_left, top_right, bottom_right])
            
        rows, cols= [512,512]
        pts1=np.float32(np.array([[0,0],[0,rows],[cols,0],[cols,rows]]))
        pts2=np.float32(ordered_points)
        M = cv2.getPerspectiveTransform(pts2, pts1)
        transform=np.array(M,dtype=np.float32)
        cuadro = cv2.warpPerspective(image, transform, (cols, rows))
        cuadros.append(cuadro)  # Append the cleaned cuadro

    return cuadros


def calculate_similarity(histograms, descriptor, labels, K, similarity_measure, quantization, structure, level, mask, folder):
    """Calculate mAP@K for a given similarity measure and descriptor."""
    measures = ComputeSimilarity()
    top_K = []

    for img in sorted(os.listdir(folder)):
        image_path = os.path.join(folder, img)
        mask_path = os.path.join(MASK_FOLDER, img.split('.')[0] + ".png")

        # Check if the file is an image
        if not img.endswith(".jpg"):
            continue
        
        try:
            image = cv2.imread(image_path)

            if mask is not None:
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                pictures = detect_pictures(image, mask)  # Detect pictures (regions of interest)

            else:
                pictures = [image]

        except Exception as e:
            print(f"Error processing image {img}: {e}")
            continue
        
        img_top_K = []

        for pict in pictures:
            try:
                histogram = descriptor.describe(pict, structure, quantization)
            except Exception as e:
                print(f"Error processing image {img}: {e}")
                continue

            if structure == 'DCT' or structure == 'LBP' or structure == 'DCT_simple':
                if similarity_measure == "intersection":
                    similarities = {key: measures.histogramIntersection(np.array(histogram,dtype=np.float32).flatten(), 
                                                                        np.array(value['descriptor'],dtype=np.float32).flatten()) for key, value in histograms.items()}
                    reverse = True
                elif similarity_measure == "bhattacharyya":
                    similarities = {key: measures.bhattacharyyaDistance(np.array(histogram,dtype=np.float32).flatten(), 
                                                                        np.array(value['descriptor'],dtype=np.float32).flatten()) for key, value in histograms.items()}
                    reverse = True
                elif similarity_measure == "Chisqr":
                    similarities = {key: measures.histogramChisqr(np.array(histogram,dtype=np.float32).flatten(), 
                                                                        np.array(value['descriptor'],dtype=np.float32).flatten()) for key, value in histograms.items()}
                    reverse = True
                elif similarity_measure == "Correl":
                    similarities = {key: measures.histogramCorrel(np.array(histogram,dtype=np.float32).flatten(), 
                                                                        np.array(value['descriptor'],dtype=np.float32).flatten()) for key, value in histograms.items()}
                    reverse = False
                elif similarity_measure == "kullback":
                    similarities = {key: kullback_leibler_divergence(np.array(histogram,dtype=np.float32).flatten(), 
                                                                        np.array(value['descriptor'],dtype=np.float32).flatten()) for key, value in histograms.items()}
                    reverse = False
                elif similarity_measure=="euclidean":
                    similarities = {key: sum(pow(abs(np.array(np.array(value['descriptor'],dtype=np.float32).flatten()-np.array(histogram,dtype=np.float32).flatten(),
                                                                dtype=np.float32).flatten()),2)) for key, value in histograms.items()}
                    reverse = False
                elif similarity_measure=="hellinger":
                    similarities = {key: np.sum(np.sqrt(np.multiply(np.array(value['descriptor'],dtype=np.float32).flatten(),
                                                                        np.array(histogram,dtype=np.float32).flatten()))) for key, value in histograms.items()}
                    reverse = True

                top_k = [k for k, v in sorted(similarities.items(), key=lambda item: item[1], reverse=reverse)][:K]
                top_k_numbers = [int(filename.split('.')[0].split('_')[-1]) for filename in top_k]
                img_top_K.append(top_k_numbers)
        
        top_K.append(img_top_K)
    return top_K  # Return top K results (list of lists)


def process_similarity_measures(histograms, descriptor, labels, quantization, structure, k_val, method_folder, measure, mask, images_folder=qsd_folder):
    """Process all combinations of similarity measures for a single descriptor."""

    if structure == 'DCT' or structure =="LBP" or structure == 'DCT_simple':
        
        top_K = calculate_similarity(histograms, descriptor, labels, k_val, measure, quantization, structure, None, mask=mask, folder=images_folder)
        map_k = mapk(labels, top_K, k_val)
        print(f"mAP@{k_val} for quantization {quantization}, {structure}, {descriptor.color_space} and {measure}: {map_k}")

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
    final_image = None
    if qsd_folder == DENOISED_IMAGES_2:
        print("Removing background")

        if not os.path.exists(NO_BG_FOLDER):
            os.makedirs(NO_BG_FOLDER)
        
        # Initialize I/O and metrics
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

                # Apply denoising
                linear_denoiser = LinearDenoiser(image)
                denoise_image = linear_denoiser.medianFilter(5)
                
                # Initialize background removal process
                background = CalculateBackground(denoise_image)
                mask_contours = background.process_frames()
                
                # Perform final morphological operations to clean up the mask
                cleaned_mask = background.morphological_operations_cleanse(mask_contours)
                
                # Find all contours in the mask
                contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Filter contours by a minimum area
                large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 10000]
                
                # Sort contours by area in descending order
                large_contours = sorted(large_contours, key=cv2.contourArea, reverse=True)
                
                # Select only the largest contours (e.g., the 2 largest ones)
                largest_contours = large_contours[:2]
                
                # Create a new mask with only the selected contours
                final_image = np.zeros(cleaned_mask.shape, dtype=np.uint8)
                for cnt in largest_contours:
                    cv2.drawContours(final_image, [cnt], -1, 255, thickness=cv2.FILLED)

                # Save the mask to MASK_FOLDER
                if not os.path.exists(MASK_FOLDER):
                    os.makedirs(MASK_FOLDER)
                cv2.imwrite(os.path.join(MASK_FOLDER, image_name.split('.')[0] + ".png"), final_image)

                # Load ground truth and compute metrics if available
                gt = cv2.imread(os.path.join(QSD2_W3_FOLDER, image_name[:-4] + ".png"), cv2.IMREAD_GRAYSCALE)
                if gt is not None:
                    intersection = np.logical_and(gt, final_image)
                    union = np.logical_or(gt, final_image)
                    iou_score = np.sum(intersection) / np.sum(union) if np.sum(union) > 0 else 0
                    iou_scores.append(iou_score)

                    precision, recall, f1 = compute_precision_recall_f1(gt, final_image)
                    precisions.append(precision)
                    recalls.append(recall)
                    f1s.append(f1)

                # Save the denoised image without background to NO_BG_FOLDER
                output_path = os.path.join(NO_BG_FOLDER, image_name.split('.')[0] + ".png")
                cv2.imwrite(output_path, image)

        qsd_folder = NO_BG_FOLDER
        # Compute mean metrics if they exist
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
    if qsd_folder == 'qsd1_w3':
        qsd_folder = QSD1_W3_FOLDER
        denoise_images = DENOISED_IMAGES_1
    elif qsd_folder == 'qsd2_w3':
        qsd_folder = QSD2_W3_FOLDER
        denoise_images = DENOISED_IMAGES_2

    if not os.path.exists(RESULTS_FOLDER):
        os.makedirs(RESULTS_FOLDER)
    
    if not os.path.exists(denoise_images):
        os.makedirs(denoise_images)
      
    if len(os.listdir(denoise_images)) == 0:
        denoiseAll(qsd_folder, denoise_images)

    histograms = load_histograms(structure, TextureDescriptor(colorspace), BBDD_FOLDER, quantization)
    # Attempt to load the ground truth labels, if the file exists
    labels = None
    try:
        with open(GT_CORRESPS_FILE, 'rb') as f:
            labels = pickle.load(f)
        print("Ground truth labels loaded successfully from", GT_CORRESPS_FILE)
    except FileNotFoundError:
        print(f"Warning: Ground truth file {GT_CORRESPS_FILE} not found. Continuing without labels.")

    _, mask = background_images(denoise_images)
    # After all masks have been applied and evaluated, process similarity with the background-removed images
    print("Processing similarity using method:", colorspace, "structure:", structure)
    process_similarity_measures(histograms, TextureDescriptor(colorspace), labels, quantization, structure, k_val=1, mask=cv2.bitwise_not(mask), measure=measure, method_folder=METHOD1_FOLDER, images_folder=denoise_images)
                