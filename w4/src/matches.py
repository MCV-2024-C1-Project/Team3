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
            
        else:
            # print(f"No descriptors for {filename}")
            keypoints = []  # Empty list for keypoints
            descriptors = np.array([])  # Placeholder for no descriptors
        
        image_paths.append(os.path.join(BBDD_FOLDER, filename))
        database_keypoints.append(keypoints)
        database_descriptors.append(descriptors)

    return image_paths, database_keypoints, database_descriptors


# Función para hacer el match entre los descriptores de la imagen de consulta y las de la base de datos
def match_with_database(query_descriptors, database_descriptors, ratio_thresh=0.7):
    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    matches_per_image = []
    
    for db_des in database_descriptors:
        if db_des.size == 0:  # Check if the descriptor array is empty
            matches_per_image.append(0)  # Indicate no matches
            continue
        # Proceed with matching if descriptors are available
        matches = flann.knnMatch(query_descriptors, db_des, k=2)
        good_matches = [m for m, n in matches if m.distance < ratio_thresh * n.distance]
        matches_per_image.append(len(good_matches))
    
    return matches_per_image

# Función principal que realiza la coincidencia con todas las imágenes de la carpeta
def find_matches_in_database(query_image, descriptor, ratio_thresh=0.7, match_threshold=5, top_k=10):
    """
    Function to find matches for the query image in the database and return top_k results.

    Parameters:
    - query_image: Query image.
    - descriptor: The type of descriptor to use ("sift", "orb", "akaze").
    - ratio_thresh: Threshold for Lowe's ratio test in feature matching.
    - match_threshold: Minimum matches required to consider an image as a match.
    - top_k: Number of top results to return.

    Returns:
    - List of top_k matches in the format [[index1, index2, ...], ...] or [[-1]] if no match.
    """
    if query_image.shape[0] > 2000 or query_image.shape[1] > 2000:
        print("resize")
        query_image = cv2.resize(query_image, (int(query_image.shape[1] / 2), int(query_image.shape[0] / 2)))
    _, query_des = get_keypoints_descriptors(query_image, descriptor)
    if query_des is None:
        return [[-1]]

    # Load the database images and get their keypoints and descriptors
    image_paths, db_keypoints, db_descriptors = load_database_images(descriptor)
    
    # # Ensure descriptor types are compatible
    # if descriptor == "akaze" or descriptor == "orb":
    #     # AKAZE and ORB descriptors are binary, ensure they are in uint8 format
    #     if query_des.dtype != np.uint8:
    #         query_des = query_des.astype(np.uint8)
    #     db_descriptors = [des.astype(np.uint8) for des in db_descriptors if des is not None]
    #     bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    # else:  # Use FLANN for SIFT which produces float32 descriptors
    #     if query_des.dtype != np.float32:
    #         query_des = query_des.astype(np.float32)
    #     db_descriptors = [des.astype(np.float32) for des in db_descriptors if des is not None]
    #     index_params = dict(algorithm=1, trees=5)
    #     search_params = dict(checks=50)
    #     bf = cv2.FlannBasedMatcher(index_params, search_params)
    # Perform matching between query descriptors and each database image's descriptors

    # Perform matching between query descriptors and each database image's descriptors
    # match_counts = []
    # for des in db_descriptors:
    #     if des is None or len(des) < 5:
    #         match_counts.append(0)
    #         continue
        
    #     # Perform KNN matching
    #     matches = bf.knnMatch(query_des, des, k=2)
    #     good_matches = []
    #     for m, n in matches:
    #         if m.distance < ratio_thresh * n.distance:
    #             good_matches.append(m)
        
    #     match_counts.append(len(good_matches))
    match_counts = match_with_database(query_des, db_descriptors, ratio_thresh)

    # Collect matches and filter by threshold
    results = []
    for idx, (path, count) in enumerate(zip(image_paths, match_counts)):
        if count >= match_threshold:
            results.append((idx, count))  # Store index and count if it meets the threshold
        else:
            results.append((idx, -1))  # Mark as unknown

    # Sort results by match count in descending order, ignoring unknowns (-1)
    sorted_results = sorted([r for r in results if r[1] != -1], key=lambda x: x[1], reverse=True)

    # Extract the top_k results based on match count, or return [[-1]] if no valid matches
    if sorted_results:
        top_k_indices = [idx for idx, _ in sorted_results[:top_k]]
    else:
        top_k_indices = [-1]

    print(f"Top {top_k} matches:", top_k_indices)
    return [top_k_indices]


if  __name__ == "__main__":
    # Cargar la imagen de consulta en escala de grises
    image = cv2.imread('./data/qsd1_w4/00026.jpg', cv2.IMREAD_GRAYSCALE)
    results = find_matches_in_database(image, "akaze")
    