# similarity.
# py - Task 2
import cv2
import sys
import os
import numpy as np
from scipy.spatial import distance as dist
from utils.preprocessing import normalizeHistogram


class ComputeSimilarity:
    
    # Function to concatenate histograms for multi-dimensional histograms (3D)
    def concatenateHistograms(self, h1, h2):
        # Normalize histograms
        h1_normalized = [normalizeHistogram(c) for c in h1]
        h2_normalized = [normalizeHistogram(c) for c in h2]

        # Concatenate histograms for each channel into a single 1D array
        h1_concat = np.concatenate(h1_normalized).ravel()
        h2_concat = np.concatenate(h2_normalized).ravel()

        return h1_concat, h2_concat
    
    # Histogram Intersection implementation (1D)
    def histogramIntersectionGrayScale(self,h1, h2):
        return cv2.compareHist(h1, h2, cv2.HISTCMP_INTERSECT)
    
    # Histogram Intersection implementation (3D)
    # Concatenate and compute the similarity
    def histogramIntersection(self,h1, h2):
        h1_concat, h2_concat = self.concatenateHistograms(h1, h2)
        tot_similarity= self.histogramIntersectionGrayScale(h1_concat, h2_concat)
        
        return tot_similarity
    
    # Bhattacharyya distance implementation (1D)
    def bhattacharyyaDistanceGrayScale(self,h1, h2):

        return cv2.compareHist(h1, h2, cv2.HISTCMP_BHATTACHARYYA)

    # Bhattacharyya distance implementation (3D)
    # Concatenate and compute the similarity
    def bhattacharyyaDistance(self,h1, h2):

        h1_concat, h2_concat = self.concatenateHistograms(h1, h2)
        tot_similarity= self.bhattacharyyaDistanceGrayScale(h1_concat, h2_concat)
        
        return tot_similarity
    
    # Canberra distance implementation (1D)
    def canberraDistanceGrayScale(self,h1, h2):
        return dist.canberra(h1, h2)

    # Canberra distance implementation (3D)
    # Concatenate and compute the similarity
    def canberraDistance(self,h1, h2):

        h1_concat, h2_concat = self.concatenateHistograms(h1, h2)
        tot_similarity= self.canberraDistanceGrayScale(h1_concat, h2_concat)
        
        return tot_similarity



