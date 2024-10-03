import numpy as np

def normalizeData(data, min_val, max_val):
    # Convertimos la lista de listas a un array de numpy
    data_array = np.array(data, dtype=np.float32)
    
    # Normalizamos los datos al rango de 0 a 1
    normalized_data = (data_array - min_val) / (max_val - min_val)
    
    return normalized_data