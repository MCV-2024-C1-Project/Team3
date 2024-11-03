import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import cv2
import numpy as np
import pickle
import argparse
from keypoints import akaze_detector, orb_detector, sift_detector
from denoiser import denoiseAll, LinearDenoiser
from background_removal import CalculateBackground
from similarity import ComputeSimilarity
from matches import find_matches_in_database
from evaluation.average_precision import mapk

# Argument parser setup
parser = argparse.ArgumentParser(description='Process image folder for keypoints detection.')
parser.add_argument('images_folder', type=str, help='Path to the image folder (e.g., ./data/qsd1_w4)')
parser.add_argument('detector', type=str, help='Keypoint detector to use (e.g., harris, orb, sift)')
args = parser.parse_args()

# Define input folder based on argument
qsd_folder = args.images_folder
detector_type = args.detector

# Constants for paths
DATA_FOLDER = './data'
RESULTS_FOLDER = './results'
BBDD_FOLDER = os.path.join(DATA_FOLDER, 'BBDD')
MASK_FOLDER =  os.path.join(DATA_FOLDER, 'masks')
GT_CORRESPS_FILE = os.path.join(DATA_FOLDER, qsd_folder , 'gt_corresps.pkl')

METHOD1_FOLDER = os.path.join(RESULTS_FOLDER, 'method1')  # Output folder for method1


def keypoints_to_serializable(keypoints):
    """Convert cv2.KeyPoint objects to a serializable format."""
    return [{
        'pt': kp.pt,                # Coordenates of the keypoint
        'size': kp.size,            # Size of the keypoint
        'angle': kp.angle,          # Angle of the keypoint
        'response': kp.response,    # Response of the keypoint
        'octave': kp.octave,        # Octave of the keypoint
        'class_id': kp.class_id     # ID of the keypoint
    } for kp in keypoints]


def detect_keypoints(detector_type, image):
    """Detect keypoints based on the selected detector type."""
    if detector_type == 'akaze':
        return akaze_detector(image)
    elif detector_type == 'orb':
        return orb_detector(image)
    elif detector_type == 'sift':
        return sift_detector(image)
    else:
        raise ValueError(f"Detector type {detector_type} not recognized.")


def process_images(folder_path, detector_type):
    """Process each image in the folder to detect keypoints and descriptors."""
    keypoints_data = {}
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(".jpg"):
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error loading image {image_path}")
                continue
            
            # Detect keypoints
            keypoints, descriptors = detect_keypoints(detector_type, image)
            serializable_keypoints = keypoints_to_serializable(keypoints)  # Convert keypoints to serializable format
            keypoints_data[filename] = {
                'keypoints': serializable_keypoints,
                'descriptors': descriptors.tolist() if descriptors is not None else None  # Convert descriptors to list
            }
            
            # Draw keypoints for visualization
            if descriptors is not None:
                output_image = cv2.drawKeypoints(image, keypoints, None, color=(0, 255, 0))
            else:  # Draw keypoints without descriptors
                output_image = image
                for kp in keypoints:
                    cv2.circle(output_image, (int(kp.pt[0]), int(kp.pt[1])), 3, (0, 255, 0), -1)

            # Save the visualization of keypoints
            cv2.imwrite(os.path.join(RESULTS_FOLDER, detector_type, f"keypoints_{detector_type}_{filename}"), output_image)
    
    return keypoints_data

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
    print("Removing background")
    
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
            denoise_image = linear_denoiser.medianFilter(3)
            
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
            print("mask saved to", {os.path.join(MASK_FOLDER, image_name.split('.')[0] + ".png")})
            # Load ground truth and compute metrics if available
            gt_path = os.path.join(qsd_folder, image_name[:-4] + ".png")
            if os.path.exists(gt_path):
                gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            else:
                gt = None
                
            if gt is not None:
                intersection = np.logical_and(gt, final_image)
                union = np.logical_or(gt, final_image)
                iou_score = np.sum(intersection) / np.sum(union) if np.sum(union) > 0 else 0
                iou_scores.append(iou_score)

                precision, recall, f1 = compute_precision_recall_f1(gt, final_image)
                precisions.append(precision)
                recalls.append(recall)
                f1s.append(f1)

    # Compute mean metrics if they exist
    if iou_scores:
        print(f"Mean IoU: {np.mean(iou_scores)}")
    if precisions:
        print(f"Mean Precision: {np.mean(precisions)}")
    if recalls:
        print(f"Mean Recall: {np.mean(recalls)}")
    if f1s:
        print(f"Mean F1 Score: {np.mean(f1s)}")
            
    # return qsd_folder, final_image
    
def detect_pictures(image, mask):
    """Detect possible cuadros (regions of interest) in the image using contours."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cuadros = []
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
    
def calculate_similarity(descriptor, K, folder):
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

            # if mask is not None:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            pictures = detect_pictures(image, mask)  # Detect pictures (regions of interest)

            # else:
            # pictures = [image]

        except Exception as e:
            print(f"E processing image {img}: {e}")
            continue
        
        img_top_K = []

        # Process each region of interest (picture) using find_matches_in_database
        for pict in pictures:
            try:
                # Use find_matches_in_database instead of calculating similarity directly
                matches = find_matches_in_database(
                    query_image=pict,
                    descriptor=descriptor,
                    top_k=K
                )
                # Append the top matches for this picture to img_top_K
                img_top_K.extend(matches)

            except Exception as e:
                print(f"Error processing region in image {img}: {e}")
                continue
        
        top_K.append(img_top_K)
    
    # print(top_K)
    return top_K  # Return top K results (list of lists)


def process_similarity_measures(descriptor, labels, k_val, method_folder, images_folder=qsd_folder):
    """Process all combinations of similarity measures for a single descriptor."""

        
    if labels is not None:
        top_K = calculate_similarity(descriptor, k_val, folder=images_folder)
        map_k = mapk(labels, top_K, k_val)
        print(f"mAP@{k_val} for {descriptor}: {map_k}")
    else:
        top_K = calculate_similarity(descriptor, k_val, folder=images_folder)

    if not os.path.exists(method_folder):
        os.makedirs(method_folder)

    pkl_output_path = os.path.join(method_folder, 'result.pkl')
    with open(pkl_output_path, 'wb') as f:
        pickle.dump(top_K, f)

    print(f"Results saved to {pkl_output_path}")
     

if __name__ == '__main__':
    if not os.path.exists(RESULTS_FOLDER):
        os.makedirs(RESULTS_FOLDER)
    
    if not os.path.exists(os.path.join(RESULTS_FOLDER, detector_type)):
        os.makedirs(os.path.join(RESULTS_FOLDER, detector_type))
        keypoints_data = process_images(BBDD_FOLDER, detector_type)
    
        # Save keypoints and descriptors data with method-specific filename
        pkl_output_path = os.path.join(RESULTS_FOLDER, f'keypoints_{detector_type}.pkl')
        with open(pkl_output_path, 'wb') as f:
            pickle.dump(keypoints_data, f)
        print(f"Keypoints and descriptors saved to {pkl_output_path}")
    else:
        print(f"Keypoints and descriptors already in {os.path.join(RESULTS_FOLDER, detector_type)}")
    
    labels = None
    try:
        with open(GT_CORRESPS_FILE, 'rb') as f:
            labels = pickle.load(f)
        print("Ground truth labels loaded successfully from", GT_CORRESPS_FILE)
    except FileNotFoundError:
        print(f"Warning: Ground truth file {GT_CORRESPS_FILE} not found. Continuing without labels.")

    # Process QSD images and compute similarity
    
    final_folder = os.path.join(DATA_FOLDER, qsd_folder)
    background_images(final_folder)
    print("Processing similarity using method:", detector_type)
    process_similarity_measures(detector_type, labels, k_val=1, method_folder=METHOD1_FOLDER, images_folder=final_folder)
     