import cv2
import numpy as np
import matplotlib.pyplot as plt

class CalculateBackground():
    def __init__(self, image):
        self.image = image

    def display_image(self, img, title):
        """Utility function to display images."""
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(title)
        plt.axis('off')
        plt.show()
    
    def detect_contours_with_gradients(self, output_dir='./data/qsd2_w2/'):
        """Detect contours in the image using gradient-based edge detection."""
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # Aplicar operador Sobel para calcular los gradientes en x y y
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=7)  # Gradiente en x
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=7)  # Gradiente en y

        # Calcular la magnitud total del gradiente
        magnitude = cv2.magnitude(grad_x, grad_y)

        # Normalizar la magnitud para que esté entre 0 y 255
        magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

        # Convertir la magnitud a tipo uint8
        magnitude = np.uint8(magnitude)

        # Aplicar un umbral para obtener los bordes binarios
        _, edges = cv2.threshold(magnitude, 40, 255, cv2.THRESH_BINARY)

        # Aplicar una dilatación para cerrar brechas entre bordes
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # Kernel de 5x5 para operaciones morfológicas
        # edges = cv2.dilate(edges, kernel, iterations=2)  # Dilatar los bordes
        # edges = cv2.erode(edges, kernel, iterations=2)  # Dilatar los bordes
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)  # Aplicar un cierre para unir bordes

        # Guardar la imagen de los bordes detectados
        cv2.imwrite('./data/qsd2_w2/edges.jpg', edges)

        # Encontrar los contornos
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        possible_frames = []
        contour_image = self.image.copy()

        mask_contours = np.zeros(self.image.shape[:2], dtype=np.uint8)  # Máscara para los contornos

        for cnt in contours:
            possible_frames.append(cnt)
            cv2.drawContours(mask_contours, [cnt], -1, 255, -1)  # Dibujar el contorno cerrado en la máscara
            cv2.drawContours(contour_image, [cnt], -1, (0, 255, 0), 5)  # Dibujar en la imagen de contorno
            # perimeter = cv2.arcLength(cnt, True)
            # approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)  # Ajustar para mayor precisión

            # # Verificar si el contorno está cerrado
            # is_closed = cv2.isContourConvex(approx)  # Verificar si el contorno es convexo
            # if is_closed and len(approx) == 4:  # Detectar rectángulos cerrados
            #     possible_frames.append(approx)
            #     cv2.drawContours(mask_contours, [approx], -1, 255, -1)  # Dibujar el contorno cerrado en la máscara
            #     cv2.drawContours(contour_image, [approx], -1, (0, 255, 0), 5)  # Dibujar en la imagen de contorno

        # Guardar la imagen con los contornos detectados
        cv2.imwrite('./data/qsd2_w2/contour_image.jpg', contour_image)
        cv2.imwrite('./data/qsd2_w2/mask_contours.jpg', mask_contours)

        return mask_contours

    def process_frames(self):
        """Process only contours (frames) detection."""
        mask_contours = self.detect_contours_with_gradients()
        
        cv2.imwrite('./data/qsd2_w2/combined_mask.jpg', mask_contours)

        return mask_contours

    def apply_mask(self, mask):
        """Apply the mask to the original image to extract the foreground."""
        background_removed = cv2.bitwise_and(self.image, self.image, mask=mask)
        return background_removed

    def morphological_operations_cleanse(self, final_mask):
        """Apply morphological operations to clean the mask."""
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (100, 1))
        cleaned_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel)
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20)))
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20)))
        
        return cleaned_mask

if __name__ == "__main__":
    # Cargar la imagen
    image = cv2.imread('./data/qsd2_w3/00022.jpg')
    background = CalculateBackground(image)

    # Detectar contornos
    mask_contours = background.process_frames()

    # Aplicar la máscara
    foreground = background.apply_mask(mask_contours)

    # Realizar operaciones morfológicas finales para limpiar la máscara
    cleaned_mask = background.morphological_operations_cleanse(mask_contours)
    cv2.imwrite('./data/qsd2_w2/cleaned_mask.jpg', cleaned_mask)
    final_image = background.apply_mask(cleaned_mask)

    # Guardar la imagen final
    cv2.imwrite('./data/qsd2_w2/final_image.jpg', final_image)
