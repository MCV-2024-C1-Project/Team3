import os
import sys
import cv2
import numpy as np
import pickle
import argparse
from keypoints import akaze_detector, orb_detector, sift_detector

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
