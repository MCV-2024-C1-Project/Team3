# similarity.py - Task 2
import cv2
from utils.preprocessing import normalizeData


class ComputeSimilarity:
    
    def histogramIntersectionGrayScale(h1, h2):
        return cv2.compareHist(h1, h2, cv2.HISTCMP_INTERSECT)
    
    def histogramIntersection(h1, h2):
        similarity_values = [
            ComputeSimilarity.histogramIntersectionGrayScale(
                normalizeData(c1,0,255), 
                normalizeData(c2,0,255)) for c1, c2 in zip(h1, h2)]
        tot_similarity = sum(similarity_values) / len(similarity_values)

        return tot_similarity
    
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



