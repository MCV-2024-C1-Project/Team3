# similarity.
# py - Task 2
import cv2
import sys
import os
from scipy.spatial import distance as dist
import numpy as np

from utils.preprocessing import normalizeHistogram


class ComputeSimilarity:
    
    # Histogram Intersection implementation (1D)
    def histogramIntersectionGrayScale(self,h1, h2):
        return cv2.compareHist(h1, h2, cv2.HISTCMP_INTERSECT)
    
    # Histogram Intersection implementation (3D)
    # First normalize the histograms, and then compute the similarity for each dimension of histogram
    # Then, the average of each value is done.
    def histogramIntersection(self,h1, h2):
        if np.array(h1).ndim==3:
            similarity_values = [
                self.histogramIntersectionGrayScale(
                    normalizeHistogram(c1), 
                    normalizeHistogram(c2)) for c1, c2 in zip(h1, h2)]
        else:
            similarity_values = [
            self.histogramIntersectionGrayScale(
                normalizeHistogram(h1), 
                normalizeHistogram(h2))]
        tot_similarity = sum(similarity_values) / len(similarity_values)

        return tot_similarity
    
    # Bhattacharyya distance implementation (1D)
    def bhattacharyyaDistanceGrayScale(self,h1, h2):

        return cv2.compareHist(h1, h2, cv2.HISTCMP_BHATTACHARYYA)

    # Bhattacharyya distance implementation (3D)
    # First normalize the histograms, and then compute the similarity for each dimension of histogram
    # Then, the average of each value is done.
    def bhattacharyyaDistance(self,h1, h2):

        similarity_values = [
            self.bhattacharyyaDistanceGrayScale(
                normalizeHistogram(c1), 
                normalizeHistogram(c2)) for c1, c2 in zip(h1, h2)]
        tot_similarity = sum(similarity_values) / len(similarity_values)

        return tot_similarity
    
    # Canberra distance implementation (1D)
    def canberraDistanceGrayScale(self,h1, h2):
        return dist.canberra(h1, h2)

    # Canberra distance implementation (3D)
    # First normalize the histograms, and then compute the similarity for each dimension of histogram
    # Then, the average of each value is done.
    def canberraDistance(self,h1, h2):
        if np.array(h1).ndim==3:
            similarity_values = [
                self.canberraDistanceGrayScale(
                    normalizeHistogram(c1), 
                    normalizeHistogram(c2)) for c1, c2 in zip(h1, h2)]
            
        else:
            similarity_values = [
                self.canberraDistanceGrayScale(
                    normalizeHistogram(h1), 
                    normalizeHistogram(h2))]
        tot_similarity = sum(similarity_values) / len(similarity_values)

        return tot_similarity



