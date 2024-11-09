import cv2
import numpy as np

def akaze_detector(image):
    """Detect keypoints and compute descriptors using AKAZE."""
    akaze = cv2.AKAZE_create(
        descriptor_type=cv2.AKAZE_DESCRIPTOR_KAZE,  # Descriptor type:
        # Options: cv2.AKAZE_DESCRIPTOR_MLDB (binary), cv2.AKAZE_DESCRIPTOR_KAZE (floating-point).
        # Default: cv2.AKAZE_DESCRIPTOR_MLDB. Binary descriptors are faster, while KAZE provides higher precision.
        
        descriptor_size=0,  # Descriptor size:
        # Default: 0 (full size). Adjustable for MLDB descriptors to reduce size and improve speed.
        
        threshold=0.001,  # Detection threshold:
        # Typical range: 0.001 - 0.01. Default: 0.001. Controls sensitivity for feature detection:
        # higher values reduce keypoints; lower values increase keypoints.
        
        nOctaves=4,  # Number of octaves in the scale pyramid:
        # Default: 4. Higher values improve robustness to scale changes but slow down processing.
        
        nOctaveLayers=4  # Number of layers per octave:
        # Default: 4. Increasing this value enhances precision but reduces speed.
    )
    keypoints, descriptors = akaze.detectAndCompute(image, None)
    return keypoints, descriptors

def orb_detector(image):
    """Detect keypoints and compute descriptors using ORB."""
    orb = cv2.ORB_create(
        nfeatures=850,  # Maximum number of keypoints to detect:
        # Default: 500. Limits the maximum number of detected keypoints for faster processing.

        scaleFactor=1.2,  # Scale factor between levels of the pyramid:
        # Default: 1.2. Typical range: 1.2 - 1.5. Higher values reduce the number of levels and improve speed,
        # but ORB becomes less robust to scale changes.
        
        nlevels=32,  # Number of levels in the scale pyramid:
        # Default: 8. More levels provide greater precision in detection but slow down processing.

        edgeThreshold=35,  # Edge threshold to avoid keypoints near image edges:
        # Default: 31. Typical range: 10 - 50. Lower values capture more points near edges,
        # but could include low-interest keypoints.

        patchSize=40,  # Patch size for computing descriptors:
        # Default: 31. Typical range: 15 - 50. Increasing captures more context around each keypoint,
        # though it increases computational cost.
        
        WTA_K=4,  # Number of points to compare in each BRIEF bin:
        # Default: 2. Options: 2, 3, or 4. Higher values are more robust but slower.

        scoreType=cv2.ORB_HARRIS_SCORE  # Keypoint scoring method:
        # Options: cv2.ORB_HARRIS_SCORE or cv2.ORB_FAST_SCORE. Default: ORB_HARRIS_SCORE.
        # Using FAST increases speed at the cost of some robustness.
    )
    keypoints, descriptors = orb.detectAndCompute(image, None)
    return keypoints, descriptors

def sift_detector(image):
    """Detect keypoints and compute descriptors using SIFT."""
    sift = cv2.SIFT_create(
        nfeatures=500,  # Maximum number of keypoints to detect:
        # Default: 0 (no limit). Limits the number of keypoints for better performance.

        contrastThreshold=0.04,  # Contrast threshold:
        # Default: 0.04. Typical range: 0.01 - 0.1. Higher values reduce keypoints in low-contrast areas,
        # useful in complex or noisy images.

        edgeThreshold=10,  # Threshold to remove edge responses:
        # Default: 10. Typical range: 5 - 20. Higher values reduce false positives on strong edges,
        # but might remove keypoints in textured areas.

        sigma=2.0  # Gaussian width for scale-space pyramid:
        # Default: 1.6. Typical range: 1.2 - 2.0. Higher values produce smoother descriptors, 
        # improving noise robustness at the cost of processing speed.
    )
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return keypoints, descriptors

def surf_detector(image):
    """Detect keypoints using FAST and compute descriptors using ORB."""
    fast = cv2.FastFeatureDetector_create(
        threshold=10,  # Threshold for the FAST detector.
        nonmaxSuppression=True,  # Non-maximum suppression flag.
        type=cv2.FAST_FEATURE_DETECTOR_TYPE_9_16  # Type of FAST detector.
    )
    keypoints = fast.detect(image, None)
    orb =orb = cv2.ORB_create(
        nfeatures=750,  # Maximum number of keypoints to detect:
        # Default: 500. Limits the maximum number of detected keypoints for faster processing.

        scaleFactor=1.2,  # Scale factor between levels of the pyramid:
        # Default: 1.2. Typical range: 1.2 - 1.5. Higher values reduce the number of levels and improve speed,
        # but ORB becomes less robust to scale changes.
        
        nlevels=32,  # Number of levels in the scale pyramid:
        # Default: 8. More levels provide greater precision in detection but slow down processing.

        edgeThreshold=40,  # Edge threshold to avoid keypoints near image edges:
        # Default: 31. Typical range: 10 - 50. Lower values capture more points near edges,
        # but could include low-interest keypoints.

        patchSize=40,  # Patch size for computing descriptors:
        # Default: 31. Typical range: 15 - 50. Increasing captures more context around each keypoint,
        # though it increases computational cost.
        
        WTA_K=4,  # Number of points to compare in each BRIEF bin:
        # Default: 2. Options: 2, 3, or 4. Higher values are more robust but slower.

        scoreType=cv2.ORB_HARRIS_SCORE,  # Keypoint scoring method:
        # Options: cv2.ORB_HARRIS_SCORE or cv2.ORB_FAST_SCORE. Default: ORB_HARRIS_SCORE.
        # Using FAST increases speed at the cost of some robustness.

    )
    keypoints, descriptors = orb.compute(image, keypoints)
    return keypoints, descriptors
