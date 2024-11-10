import numpy as np

def apk(actual, predicted, k=10):
    """
    Computes the average precision at k.

    This function computes the average prescision at k between two lists of
    items.

    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements

    Returns
    -------
    score : double
            The average precision at k over the input lists

    """
    if isinstance(actual, int):
        actual = [actual]
        
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0


    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)

def mapk(actual, predicted, k=10):
    """
    Computes the mean average precision at k for the new format.

    This function computes the mean average precision at k between two lists
    of lists of lists of items, where the first level corresponds to images,
    the second level corresponds to regions of interest (cuadros), and the third
    level corresponds to the predicted elements for each cuadro.

    Parameters
    ----------
    actual : list
             A list of lists of lists of elements that are to be predicted
    predicted : list
                A list of lists of lists of predicted elements
    k : int, optional
        The maximum number of predicted elements

    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    scores = []
    
    # Recorrer las imágenes
    for actual_image, predicted_image in zip(actual, predicted):
        image_scores = []
        
        # Recorrer los cuadros de la imagen
        for actual_cuadro, predicted_cuadro in zip(actual_image, predicted_image):
            image_scores.append(apk(actual_cuadro, predicted_cuadro, k))
        
        # Promedio de precisión para todos los cuadros en una imagen
        scores.append(np.mean(image_scores))
    
    # Devolver el promedio de precisión entre todas las imágenes
    return np.mean(scores)


import numpy as np
from sklearn.metrics import f1_score

def f1_at_k(actual, predicted, k=10):
    """
    Computes the F1 score at k.

    This function computes the F1 score at k between two lists of
    items.

    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements

    Returns
    -------
    score : double
            The F1 score at k over the input lists

    """
    if isinstance(actual, int):
        actual = [actual]
        
    if len(predicted) > k:
        predicted = predicted[:k]
    
    # Convert to sets for F1 score calculation
    actual_set = set(actual)
    predicted_set = set(predicted)
    
    # Calculate precision, recall, and F1 score
    true_positives = len(actual_set & predicted_set)
    precision = true_positives / len(predicted_set) if predicted_set else 0
    recall = true_positives / len(actual_set) if actual_set else 0
    
    if precision + recall == 0:
        return 0.0
    
    return 2 * (precision * recall) / (precision + recall)

def mean_f1_at_k(actual, predicted, k=10):
    """
    Computes the mean F1 score at k for the new format.

    This function computes the mean F1 score at k between two lists
    of lists of lists of items, where the first level corresponds to images,
    the second level corresponds to regions of interest (cuadros), and the third
    level corresponds to the predicted elements for each cuadro.

    Parameters
    ----------
    actual : list
             A list of lists of lists of elements that are to be predicted
    predicted : list
                A list of lists of lists of predicted elements
    k : int, optional
        The maximum number of predicted elements

    Returns
    -------
    score : double
            The mean F1 score at k over the input lists
    """
    scores = []
    
    # Iterate over images
    for actual_image, predicted_image in zip(actual, predicted):
        image_scores = []
        
        # Iterate over cuadros in the image
        for actual_cuadro, predicted_cuadro in zip(actual_image, predicted_image):
            image_scores.append(f1_at_k(actual_cuadro, predicted_cuadro, k))
        
        # Average F1 score for all cuadros in an image
        scores.append(np.mean(image_scores))
    
    # Return the average F1 score across all images
    return np.mean(scores)