# similarity.py - Task 2
import cv2
import math
import numpy as np
from utils.preprocessing import normalizeData

# Bhattacharyya distance implementation (1D)
def bhattacharyyaDistanceGrayScale(h1, h2):
    return cv2.compareHist(h1, h2, cv2.HISTCMP_BHATTACHARYYA)

# Bhattacharyya distance implementation (3D)
def bhattacharyyaDistance(h1, h2):

    similarity_values = [bhattacharyyaDistanceGrayScale(normalizeData(c1,1,4), normalizeData(c2,1,4)) for c1, c2 in zip(h1, h2)]
    tot_similarity = sum(similarity_values) / len(similarity_values)

    return tot_similarity

def emdDistance(h1, h2):
    #TODO: Implement distance
    return 0


def testSimilarity():
    h1 = [[1,2,3,4], [2,1,4,3], [4,3,2,1]]
    h2 = [[2,1,4,3], [1,2,3,4], [4,3,2,1]]

    print(f"Similarity between test 1 histograms: {bhattacharyyaDistance(h1, h2)}")

    h3 = [[1,2,3,4], [2,1,4,3], [4,3,2,1]]
    h4 = [[1,2,3,4], [2,1,4,3], [4,3,2,1]]

    print(f"Similarity between test 2 histograms: {bhattacharyyaDistance(h3, h4)}")


if __name__ == "__main__":
    testSimilarity()
