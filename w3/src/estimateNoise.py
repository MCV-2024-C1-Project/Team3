import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import log10, sqrt
from skimage.metrics import structural_similarity as ssim
import os

class NoiseEstimation():
    def __init__(self, img):
        self.img = img

    def noise_estimation_std(self, threshold):
        img = self.img.copy()

        deviation = np.std(img)

        if deviation > threshold:
            return True
        else:
            return False

    def noise_estimation_fft(self, threshold):
        img = self.img.copy()
        
        # Obtener las dimensiones de la imagen
        print(img.shape)
        rows, cols = img.shape[:2]
        
        # Aplicar la FFT y trasladar el origen a la frecuencia cero al centro
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        
        # Calcular el espectro de magnitud
        magnitude_spectrum = np.log(np.abs(fshift) + 1)
        
        # Definir el rango de frecuencias altas como una banda circular desde el centro
        center_row, center_col = rows // 2, cols // 2
        
        # Crear una máscara para frecuencias altas (fuera de un radio)
        radius = min(center_row, center_col) // 4  # Ajustar el radio según necesidad
        y, x = np.ogrid[:rows, :cols]
        mask = (x - center_col)**2 + (y - center_row)**2 > radius**2
        
        # Sumar la energía de las frecuencias altas
        high_freq_energy = np.sum(magnitude_spectrum[mask])
        
        # Comparar la energía de frecuencias altas con el umbral
        if high_freq_energy > threshold:
            return True
        else:
            return False

    def noise_estimation_tv(self, threshold):
        img = self.img.copy()

        grad_x = np.roll(img, -1, axis=1) - img
        grad_y = np.roll(img, -1, axis=0) - img
        tv_norm = np.sum(np.sqrt(grad_x**2 + grad_y**2))

        if tv_norm > threshold:
            return True
        else:
            return False

    def noise_estimation_psnr(self):
        pass

    def noise_estimation_snr(self, orig_img, threshold):
        mse = np.mean((orig_img - self.img) ** 2, axis=(0, 1))  # Calcula MSE por canal

        
        max_pixel = 255.0
        # Calculamos el PSNR en cada canal y luego promediamos
        psnr_value = 20 * log10(max_pixel / sqrt(np.mean(mse)+0.001))

        if psnr_value < threshold:
            return True
        else:
            return False

    def noise_estimation_wavelet(self):
        pass

class NoiseMetric():
    def psnr(self, orig_img, denoised_img):
        # Asegúrate de que las imágenes sean del mismo tamaño
        assert orig_img.shape == denoised_img.shape, "Las dimensiones de las imágenes deben coincidir"
        
        # Si es una imagen a color (3 canales), calculamos el MSE para cada canal y promediamos
        mse = np.mean((orig_img - denoised_img) ** 2, axis=(0, 1))  # Calcula MSE por canal
        if np.mean(mse) == 0:
            return float('inf')
        
        max_pixel = 255.0
        # Calculamos el PSNR en cada canal y luego promediamos
        psnr_value = 20 * log10(max_pixel / sqrt(np.mean(mse)))
        return psnr_value

    def ssim(self, orig_img, denoised_img):
        if len(orig_img.shape) == 3:  # If the image is colored (3 channels)
            ssim_value = ssim(orig_img, denoised_img, win_size=3, multichannel=True)
        else:  # For grayscale images
            ssim_value = ssim(orig_img, denoised_img, win_size=3)
        return ssim_value


if __name__ == "__main__":

    normal_images = ["00000.jpg", "00001.jpg", "00002.jpg", "00003.jpg", "00004.jpg", "00007.jpg", "00009.jpg", "00010.jpg", "00011.jpg", "00012.jpg", "00013.jpg", "00014.jpg", "00015.jpg", "00017.jpg", "00019.jpg", "00021.jpg", "00022.jpg", "00024.jpg", "00025.jpg", "00026.jpg", "00027.jpg", "00028.jpg", "00029.jpg"]
    noisy_images = ["00005.jpg", "00006.jpg", "00008.jpg", "00016.jpg", "00018.jpg", "00020.jpg", "00023.jpg"]
    info_images = [0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0]

    std_info = []
    fft_info = []
    tv_info = []
    snr_info = []


    for image_path in os.listdir("./data/qsd1_w3"):
        if image_path.endswith(".jpg"):
            img = cv2.imread("./data/qsd1_w3/" + image_path)
            orig_img = cv2.imread("./data/non_augmented/" + image_path)

            print("NAME: ", image_path)
            print("IMAGE: ", img.shape)
            print("Original IMAGE: ", orig_img.shape)
            noise = NoiseEstimation(img)

            std_info.append(1-int(noise.noise_estimation_std(5)))
            fft_info.append(1-int(noise.noise_estimation_fft(1)))
            tv_info.append(1-int(noise.noise_estimation_tv(10)))
            snr_info.append(1-int(noise.noise_estimation_snr(orig_img, 40)))

    # compare with info_images
    print("STD INFO: ", std_info)
    print("FFT INFO: ", fft_info)
    print("TV INFO: ", tv_info)
    print("SNR INFO: ", np.sum([1 for i in range(len(snr_info)) if snr_info[i] == info_images[i]]))

    normal_img1 = cv2.imread("./data/qsd1_w3/00003.jpg")
    normal_img2 = cv2.imread("./data/non_augmented/00003.jpg")

    print("NON NOISY PSNR: ", NoiseMetric().psnr(normal_img1, normal_img2))

    noisy_img1 = cv2.imread("./data/qsd1_w3/00005.jpg")
    normal_img2 = cv2.imread("./data/non_augmented/00005.jpg")

    print("NOISY PSNR: ", NoiseMetric().psnr(noisy_img1, normal_img2))

    mean_normal_psnr = []
    mean_noisy_psnr = []

    for name in normal_images:
        normal_img1 = cv2.imread("./data/qsd1_w3/" + name)
        normal_img2 = cv2.imread("./data/non_augmented/" + name)
        mean_normal_psnr.append(NoiseMetric().psnr(normal_img1, normal_img2))

    print("MEAN NORMAL PSNR: ", np.mean(mean_normal_psnr))
    
    for name in noisy_images:
        noisy_img1 = cv2.imread("./data/qsd1_w3/" + name)
        normal_img2 = cv2.imread("./data/non_augmented/" + name)
        mean_noisy_psnr.append(NoiseMetric().psnr(noisy_img1, normal_img2))

    print("MEAN NOISY PSNR: ", np.mean(mean_noisy_psnr))