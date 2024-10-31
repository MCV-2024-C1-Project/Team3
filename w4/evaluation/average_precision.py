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