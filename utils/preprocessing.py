import numpy as np

def normalizeData(data, min_val, max_val):
    # Convertimos la lista de listas a un array de numpy
    data_array = np.array(data, dtype=np.float32)
    
    # Normalizamos los datos al rango de 0 a 1
    normalized_data = (data_array - min_val) / (max_val - min_val)
    
    return normalized_data

def normalizeHistogram(hist):
    # Normalizamos el histograma para que la suma sea 1
    hist_sum = np.sum(hist)
    if hist_sum == 0:
        return hist  # Evitamos la división por 0
    return hist / hist_sum