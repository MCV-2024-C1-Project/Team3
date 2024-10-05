# similarity.
# py - Task 2
import cv2
import sys
import os
from scipy.spatial import distance as dist


from utils.preprocessing import normalizeData
from utils.preprocessing import normalizeHistogram


class ComputeSimilarity:
    
    def histogramIntersectionGrayScale(self,h1, h2):
        return cv2.compareHist(h1, h2, cv2.HISTCMP_INTERSECT)
    
    def histogramIntersection(self,h1, h2):
        similarity_values = [
            self.histogramIntersectionGrayScale(
                normalizeHistogram(c1), 
                normalizeHistogram(c2)) for c1, c2 in zip(h1, h2)]
        tot_similarity = sum(similarity_values) / len(similarity_values)
        #if tot_similarity > 1: print(tot_similarity)
        return tot_similarity
    
    # Bhattacharyya distance implementation (1D)
    def bhattacharyyaDistanceGrayScale(self,h1, h2):
        #return dist.canberra(h1, h2)
        return cv2.compareHist(h1, h2, cv2.HISTCMP_BHATTACHARYYA)

    # Bhattacharyya distance implementation (3D)
    def bhattacharyyaDistance(self,h1, h2):

        similarity_values = [
            self.bhattacharyyaDistanceGrayScale(
                normalizeHistogram(c1), 
                normalizeHistogram(c2)) for c1, c2 in zip(h1, h2)]
        tot_similarity = sum(similarity_values) / len(similarity_values)
        #if tot_similarity > 1: print(tot_similarity)
        return tot_similarity
    
    # Bhattacharyya distance implementation (1D)
    def canberraDistanceGrayScale(self,h1, h2):
        return dist.canberra(h1, h2)

    # Bhattacharyya distance implementation (3D)
    def canberraDistance(self,h1, h2):

        similarity_values = [
            self.canberraDistanceGrayScale(
                normalizeHistogram(c1), 
                normalizeHistogram(c2)) for c1, c2 in zip(h1, h2)]
        tot_similarity = sum(similarity_values) / len(similarity_values)
        #if tot_similarity > 1: print(tot_similarity)
        return tot_similarity



