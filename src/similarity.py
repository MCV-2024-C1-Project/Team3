# similarity.py - Task 2
import cv2
import os
from src.descriptors import HistogramDescriptor
import numpy as np
from utils.preprocessing import normalizeData


class ComputeSimilarity:

    def histogramIntersection(h1, color_space):
        # Create a histogram descriptor
        descriptor = HistogramDescriptor(color_space)

        folder_path = 'data/BBDD'
        similarities = []

        # Iterate through all files in the folder data/BBDD
        for filename in os.listdir(folder_path):
            if filename.endswith(".jpg"):
                # Load and describe the current image
                image_path = os.path.join(folder_path, filename)
                image = cv2.imread(image_path)

                h2 = descriptor.describe(image)

                # Compare the histogram of the base image with the current image
                similarity = cv2.compareHist(h1, h2, cv2.HISTCMP_INTERSECT)
                similarities.append((filename, similarity))

        
        # Sort the list of similarities from highest to lowest
        similarities_sorted = sorted(similarities, key=lambda x: x[1], reverse=True)

        return similarities_sorted
    
    # Bhattacharyya distance implementation (1D)
    def bhattacharyyaDistanceGrayScale(h1, h2):
        return cv2.compareHist(h1, h2, cv2.HISTCMP_BHATTACHARYYA)

    # Bhattacharyya distance implementation (3D)
    def bhattacharyyaDistance(h1, h2):

        similarity_values = [
            ComputeSimilarity.bhattacharyyaDistanceGrayScale(
                normalizeData(c1,0,255), 
                normalizeData(c2,0,255)) for c1, c2 in zip(h1, h2)]
        tot_similarity = sum(similarity_values) / len(similarity_values)

        return tot_similarity



