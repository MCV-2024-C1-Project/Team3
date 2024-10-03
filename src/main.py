import similarity as sm
import cv2

if __name__ == '__main__':
    print('Holis :3')

    # Carga de las imágenes
    # image = cv2.imread('data/BBDD/bbdd_00198.jpg')
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image2 = cv2.imread('data/qsd1_w1/00009.jpg')
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

    # Cálculo de los histogramas
    # hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    # hist = cv2.normalize(hist, hist).flatten()
    
    hist2 = cv2.calcHist([image2], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist2 = cv2.normalize(hist2, hist2).flatten()

    # Cálculo de la similitud usando el método de intersección de histogramas
    r = sm.histogramIntersection(hist2, 'HSV')
    # Print the sorted results
    print("\nSimilarity results (from highest to lowest):")
    for filename, similarity in r:
        print(f"{filename}: {similarity}")
