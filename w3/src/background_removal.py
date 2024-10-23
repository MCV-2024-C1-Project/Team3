import cv2
import numpy as np
import matplotlib.pyplot as plt
from denoiser import LinearDenoiser

class CalculateBackground():
    def __init__(self, image):
        self.image = image

    def display_image(self, img, title):
        """Utility function to display images."""
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(title)
        plt.axis('off')
        plt.show()
        
    def detect_contours_with_laplacian(self):
        """Detect contours in the image using Laplacian edge detection."""
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # Aplicar el filtro Laplaciano para calcular el gradiente
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian = np.uint8(np.absolute(laplacian))  # Convertir a uint8 para visualizarlo

        # Aplicar un umbral para obtener los bordes binarios
        _, edges = cv2.threshold(laplacian, 30, 255, cv2.THRESH_BINARY)

        # Aplicar una dilatación para cerrar brechas entre bordes
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))  
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)  

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

        # Guardar la imagen con los contornos detectados
        cv2.imwrite('./data/qsd2_w2/contour_image.jpg', contour_image)
        cv2.imwrite('./data/qsd2_w2/mask_contours.jpg', mask_contours)

        return mask_contours
    
    def detect_contours_with_prewitt(self, output_dir='./data/qsd2_w2/'):
        """Detect contours in the image using Prewitt filter."""
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # Aplicar los operadores Prewitt en x y y
        kernelx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
        kernely = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])

        prewitt_x = cv2.filter2D(gray, -1, kernelx)  # Gradiente en x
        prewitt_y = cv2.filter2D(gray, -1, kernely)  # Gradiente en y

        # Calcular la magnitud total del gradiente
        magnitude = np.sqrt(np.square(prewitt_x) + np.square(prewitt_y))
        magnitude = np.uint8(magnitude)

        # Aplicar un umbral para obtener los bordes binarios
        _, edges = cv2.threshold(magnitude, 30, 255, cv2.THRESH_BINARY)

        # Aplicar una dilatación para cerrar brechas entre bordes
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))  
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)  

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

        # Guardar la imagen con los contornos detectados
        cv2.imwrite('./data/qsd2_w2/contour_image.jpg', contour_image)
        cv2.imwrite('./data/qsd2_w2/mask_contours.jpg', mask_contours)

        return mask_contours
    
    def detect_contours_with_canny(self):
        """Detect contours in the image using Canny edge detection."""
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # Aplicar filtro de Canny (Los umbrales pueden ajustarse según las necesidades)
        edges = cv2.Canny(gray, 10, 200)

        # Aplicar una dilatación para cerrar brechas entre bordes
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))  
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)  

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

        cv2.imwrite('./data/qsd2_w2/contour_image.jpg', contour_image)
        cv2.imwrite('./data/qsd2_w2/mask_contours.jpg', mask_contours)

        return mask_contours
    
    def fourier_transform_background_detection(self):
        """Detect background using Fourier Transform (frequency domain analysis)."""
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        
        # Apply DFT (Discrete Fourier Transform)
        dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)  # Shift zero frequency to center

        # Get magnitude spectrum
        # magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

        # Create a mask with a high-pass filter (remove low frequencies, retain high frequencies)
        rows, cols = gray.shape
        crow, ccol = rows // 2, cols // 2  # Center point
        mask = np.ones((rows, cols, 2), np.uint8)
        r = 40  # Radius of the high-pass filter
        mask[crow - r:crow + r, ccol - r:ccol + r] = 0  # Masking the center (low frequencies)

        # Apply the mask and inverse DFT
        fshift = dft_shift * mask
        f_ishift = np.fft.ifftshift(fshift)
        img_back = cv2.idft(f_ishift)
        img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

        # Normalize and threshold the result to obtain a binary mask
        img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
        _, freq_mask = cv2.threshold(np.uint8(img_back), 50, 255, cv2.THRESH_BINARY)

        # Optionally apply some morphological operations to clean the frequency mask
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        freq_mask = cv2.morphologyEx(freq_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Save the mask for visualization
        cv2.imwrite('./data/qsd2_w2/frequency_mask.jpg', freq_mask)
        
        # Encontrar los contornos en la máscara de frecuencias
        contours, _ = cv2.findContours(freq_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Crear una máscara de contornos de frecuencias
        mask_contours_frequencies = np.zeros(self.image.shape[:2], dtype=np.uint8)

        # Rellenar los contornos detectados en la máscara
        for cnt in contours:
            cv2.drawContours(mask_contours_frequencies, [cnt], -1, 255, -1)  # Rellenar contornos

        # Guardar la máscara de contornos de frecuencias
        cv2.imwrite('./data/qsd2_w2/frequency_contours_mask.jpg', mask_contours_frequencies)

        return mask_contours_frequencies

    
    def detect_contours_with_gradients(self):
        """Detect contours in the image using gradient-based edge detection."""
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2HLS)[:, :, 1]
        
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
        _, edges = cv2.threshold(magnitude, 15, 255, cv2.THRESH_BINARY)

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
        # mask_contours = self.detect_contours_with_gradients()
        # mask_contours = self.detect_contours_with_laplacian()
        # mask_contours = self.detect_contours_with_prewitt()
        # mask_contours = self.detect_contours_with_canny()
        
        edges_mask = self.detect_contours_with_gradients()
        

        # # Detect background using Fourier Transform and find contours
        # frequency_mask = self.fourier_transform_background_detection()

        # # Convert both masks to float32 for weighted sum
        # edges_mask = np.float32(edges_mask)
        # frequency_mask = np.float32(frequency_mask)

        # # Apply a weighted sum to prioritize edges over frequencies
        # weighted_combination = cv2.addWeighted(edges_mask, 0.7, frequency_mask, 0.3, 0)

        # # Threshold the combined result to get a binary mask
        # _, combined_mask = cv2.threshold(weighted_combination, 20, 255, cv2.THRESH_BINARY)

        # # Convert back to uint8
        # combined_mask = np.uint8(combined_mask)
        
        # # Guardar la máscara combinada ponderada
        # cv2.imwrite('./data/qsd2_w2/weighted_combined_mask.jpg', combined_mask)

        return edges_mask

    def apply_mask(self, mask):
        """Apply the mask to the original image to extract the foreground."""
        background_removed = cv2.bitwise_and(self.image, self.image, mask=mask)
        return background_removed

    def morphological_operations_cleanse(self, final_mask):
        """Apply morphological operations to clean the mask."""
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (120, 1))
        cleaned_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel)
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10)))
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10)))
        
        return cleaned_mask

if __name__ == "__main__":
    # Cargar la imagen
    image = cv2.imread('./data/qsd2_w3/00008.jpg')
    linear_denoiser = LinearDenoiser(image)
    
    denoise_image = linear_denoiser.medianFilter(5)
    cv2.imwrite('./data/qsd2_w2/denoise_image.jpg', denoise_image)
    background = CalculateBackground(denoise_image)

    # Detectar contornos
    mask_contours = background.process_frames()

    # Aplicar la máscara
    foreground = background.apply_mask(mask_contours)

    # Realizar operaciones morfológicas finales para limpiar la máscara
    cleaned_mask = background.morphological_operations_cleanse(mask_contours)
            
    # Encontrar todos los contornos en la máscara
    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filtrar los contornos por un área mínima
    large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 10000]
    
    # Ordenar los contornos por área en orden descendente
    large_contours = sorted(large_contours, key=cv2.contourArea, reverse=True)
    
    # Seleccionar solo los contornos más grandes (ej: 2 más grandes)
    largest_contours = large_contours[:2]
    
    # Crear una nueva máscara con solo los contornos seleccionados
    filtered_mask = np.zeros(cleaned_mask.shape, dtype=np.uint8)
    for cnt in largest_contours:
        cv2.drawContours(filtered_mask, [cnt], -1, 255, thickness=cv2.FILLED)

    cv2.imwrite('./data/qsd2_w2/cleaned_mask.jpg', filtered_mask)
    final_image = background.apply_mask(filtered_mask)

    # Guardar la imagen final
    cv2.imwrite('./data/qsd2_w2/final_image.jpg', final_image)
