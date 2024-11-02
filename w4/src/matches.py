import cv2
import numpy as np
import os
import keypoints as kp
import pickle
import matplotlib.pyplot as plt

# Constants for paths
DATA_FOLDER = './data'
RESULTS_FOLDER = './results'
BBDD_FOLDER = os.path.join(DATA_FOLDER, 'BBDD')

def display_image(img, title):
        """Utility function to display images."""
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(title)
        plt.axis('off')
        plt.show()

# Función para convertir de representación serializable a keypoints de OpenCV
def serializable_to_keypoints(serializable_keypoints):
    # print("Debug: Estructura de 'serializable_keypoints':", serializable_keypoints)
    # Convierte a KeyPoint solo si el formato es correcto
    try:
        return [cv2.KeyPoint(x=kp[0][0], y=kp[0][1], _size=kp[1], _angle=kp[2], 
                             _response=kp[3], _octave=int(kp[4]), _class_id=int(kp[5])) 
                for kp in serializable_keypoints]
    except (IndexError, TypeError, KeyError) as e:
        # print("Error al convertir los keypoints:", e)
        return []

# Función para obtener keypoints y descriptores de una imagen
def get_keypoints_descriptors(image, descriptor):
    if descriptor == "sift":
        keypoints, descriptors = kp.sift_detector(image)
    elif descriptor == "orb":
        keypoints, descriptors = kp.orb_detector(image)
    elif descriptor == "akaze":
        keypoints, descriptors = kp.akaze_detector(image)
    else:
        raise ValueError("Descriptor no reconocido. Use 'sift', 'orb', o 'akaze'.")
    return keypoints, descriptors

# Función para cargar imágenes desde una carpeta, obteniendo sus keypoints y descriptores
def load_database_images(descriptor_type):
    keypoints_data_path = os.path.join(RESULTS_FOLDER, f"keypoints_{descriptor_type}.pkl")
    if not os.path.exists(keypoints_data_path):
        raise FileNotFoundError(f"Archivo de datos de keypoints no encontrado: {keypoints_data_path}")

    # Cargar el diccionario desde el archivo .pkl
    with open(keypoints_data_path, 'rb') as f:
        keypoints_data = pickle.load(f)
    
    # Inicializar listas para los datos de salida
    image_paths = []
    database_keypoints = []
    database_descriptors = []
    
    # Extraer datos de cada imagen
    for filename, data in keypoints_data.items():
        if data['descriptors'] is not None:
            keypoints = serializable_to_keypoints(data['keypoints'])
            descriptors = np.array(data['descriptors'], dtype=np.float32)  # Convertir lista a array numpy
            image_paths.append(os.path.join(BBDD_FOLDER, filename))
            database_keypoints.append(keypoints)
            database_descriptors.append(descriptors)
        else:
            print(f"No descriptors for {filename}")

    return image_paths, database_keypoints, database_descriptors


# Función para hacer el match entre los descriptores de la imagen de consulta y las de la base de datos
def match_with_database(query_descriptors, database_descriptors, ratio_thresh=0.7):
    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    matches_per_image = []
    
    for db_des in database_descriptors:
        matches = flann.knnMatch(query_descriptors, db_des, k=2)
        good_matches = [m for m, n in matches if m.distance < ratio_thresh * n.distance]
        matches_per_image.append(len(good_matches))
    
    return matches_per_image

# Función principal que realiza la coincidencia con todas las imágenes de la carpeta
def find_matches_in_database(query_image, descriptor, ratio_thresh=0.7, match_threshold=3):
    # Obtener keypoints y descriptores de la imagen de consulta
    display_image(query_image, "original")
    _, query_des = get_keypoints_descriptors(query_image, descriptor)
    if query_des is None:
        print("No descriptors obtained")
        return None, None
    
    # Cargar las imágenes de la base de datos y obtener sus keypoints y descriptores
    image_paths, db_keypoints, db_descriptors = load_database_images(descriptor)
    
    # Realizar el match entre la imagen de consulta y cada imagen en la base de datos
    match_counts = match_with_database(query_des, db_descriptors, ratio_thresh)
    
    results = []
    for path, count in zip(image_paths, match_counts):
        if count >= match_threshold:
            results.append((path, count))  # Coincidencia válida
            print(f"{path}: {count} coincidencias")
        else:
            results.append((path, -1))  # Imagen desconocida ("unknown")
            print(f"{path}: Unknown")
        
    return results

if  __name__ == "__main__":
    # Cargar la imagen de consulta en escala de grises
    image = cv2.imread('./data/qsd1_w4/00004.jpg', cv2.IMREAD_GRAYSCALE)
    results = find_matches_in_database(image, "sift")