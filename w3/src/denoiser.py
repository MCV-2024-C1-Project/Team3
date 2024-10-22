import cv2
import numpy as np
import pywt
import os
from math import log10, sqrt
from skimage.metrics import structural_similarity as ssim

class LinearDenoiser():
    def __init__(self, image):
        self.img = image

    def boxFilter(self, kernelSize):
        img=self.img.copy()
        return cv2.blur(img, (kernelSize, kernelSize))
    
    def medianFilter(self, kernelSize):
        img = self.img.copy()
        return cv2.medianBlur(img, kernelSize)
    
    def midPointFilter(self, kernelSize):
        return True
    
    def butterworthLowPassFilter(self, d0, n):

        img = self.img.copy()

        if len(img.shape) == 3:
            # Separate the image color channels
            b, g, r = cv2.split(img)

            # Apply the filter to each channel
            b_filtered = self.butterworthGrayLowPass(b, d0, n)
            g_filtered = self.butterworthGrayLowPass(g, d0, n)
            r_filtered = self.butterworthGrayLowPass(r, d0, n)

            # Combine filtered channels
            filtered_image = cv2.merge((b_filtered, g_filtered, r_filtered))
            return filtered_image
        
        else:
            # If the image is on grayscale, apply directly
            return self.butterworthGrayLowPass(img, d0, n)
    
    def butterworthGrayLowPass(self, channelImg, d0, n):

        f = np.fft.fft2(channelImg)
        fshift = np.fft.fftshift(f)

        rows, cols = channelImg.shape

        x = np.linspace(-cols/2, cols/2, cols)
        y = np.linspace(-rows/2, rows/2, rows)
        x, y = np.meshgrid(x, y)
        distance = np.sqrt((x)**2 + (y)**2)

        butterworth_filter = 1 / (1 + (distance / d0)**(2*n))

        fshift_filtered = fshift * butterworth_filter
        f_ishift = np.fft.ifftshift(fshift_filtered)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)

        return np.uint8(np.clip(img_back, 0, 255))
    
    def nyquistFilter(self):
        return True
    
    def gaussianFilter(self, kernelSize, sigmaX):
        img = self.img.copy()
        return cv2.GaussianBlur(img, (kernelSize, kernelSize), sigmaX)
    
    def bilateralFilter(self, d, sigmaColor, sigmaSpace):
        img = self.img.copy()
        return cv2.bilateralFilter(img, d, sigmaColor, sigmaSpace)
    
    def Convolution2DFilter(self, kernelSize):
        img = self.img.copy()
        kernel = np.ones((kernelSize, kernelSize), np.float32) / (kernelSize * kernelSize)
        return cv2.filter2D(img, -1, kernel)
    
    def fftLowPassFilter(self, cutoff):

        img = self.img.copy()

        if len(img.shape) == 3:

            b, g, r = cv2.split(img)

            b_filtered = self.fftLowPassGray(b, cutoff)
            g_filtered = self.fftLowPassGray(g, cutoff)
            r_filtered = self.fftLowPassGray(r, cutoff)

            filtered_image = cv2.merge((b_filtered, g_filtered, r_filtered))
            return filtered_image
        else:
            return self.fftLowPassGray(img, cutoff)

    
    def fftLowPassGray(self, image, cutoff):

        f = np.fft.fft2(image)
        fshift = np.fft.fftshift(f)

        rows, cols = image.shape
        crow, ccol = rows // 2, cols // 2
        mask = np.zeros((rows, cols), np.uint8)
        mask[crow - cutoff:crow + cutoff, ccol - cutoff:ccol + cutoff] = 1

        fshift_filtered = fshift * mask
        f_ishift = np.fft.ifftshift(fshift_filtered)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)
        return img_back
    
    def gaussianLowPassFilter(self, sigma):

        img = self.img.copy()
        
        if len(img.shape) == 3:

            b, g, r = cv2.split(img)

            b_filtered = self.gaussianLowPassGray(b, sigma)
            g_filtered = self.gaussianLowPassGray(g, sigma)
            r_filtered = self.gaussianLowPassGray(r, sigma)

            filtered_image = cv2.merge((b_filtered, g_filtered, r_filtered))
            return filtered_image
        else:
            return self.gaussianLowPassGray(img, sigma)
    
    def gaussianLowPassGray(self, image, sigma):

        f = np.fft.fft2(image)
        fshift = np.fft.fftshift(f)

        rows, cols = image.shape

        x = np.linspace(-cols/2, cols/2, cols)
        y = np.linspace(-rows/2, rows/2, rows)
        x, y = np.meshgrid(x, y)
        gaussian_filter = np.exp(-(x**2 + y**2) / (2 * sigma**2))

        fshift_filtered = fshift * gaussian_filter
        f_ishift = np.fft.ifftshift(fshift_filtered)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)
        return img_back
    
    def waveletTransformGray(self, img, wavelet):
        coeffs = pywt.wavedec2(img, wavelet)
        coeffs_thresh = [coeffs[0]]  # Approximation coefficients

        # Apply thresholding to detail coefficients
        for detail_level in coeffs[1:]:
            cH, cV, cD = detail_level
            cH = pywt.threshold(cH, np.std(cH) / 2, mode='soft')
            cV = pywt.threshold(cV, np.std(cV) / 2, mode='soft')
            cD = pywt.threshold(cD, np.std(cD) / 2, mode='soft')
            coeffs_thresh.append((cH, cV, cD))

        # Reconstruct the image
        denoised_image = pywt.waverec2(coeffs_thresh, wavelet)
        return denoised_image

    def waveletTransformFilter(self, wavelet_type):

        img = self.img.copy()
        b, g, r = cv2.split(img)
        b_filtered = self.waveletTransformGray(b, wavelet_type)
        g_filtered = self.waveletTransformGray(g, wavelet_type)
        r_filtered = self.waveletTransformGray(r, wavelet_type)
        return cv2.merge((b_filtered, g_filtered, r_filtered))

    
class NonLinearDenoiser():
    def __init__(self, image):
        self.img = image

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

    
    metrics = NoiseMetric()

    linear_psnr_box = []
    linear_psnr_median = []
    linear_psnr_butterworth = []
    linear_psnr_gaussian = []
    linear_psnr_bilateral = []
    linear_psnr_conv = []
    linear_psnr_fft = []
    linear_psnr_gauslow = []
    linear_psnr_wavhaar = []
    linear_psnr_wavdau =  []
    linear_psnr_wavdio = []
    psnr_orig = []


    img = cv2.imread('./data/qsd1_w3/00018.jpg')
    img_orig = cv2.imread('./data/non_augmented/00018.jpg')


    linear_denoiser = LinearDenoiser(img)

    box_filter_denoised = linear_denoiser.boxFilter(3)
    median_filter_denoised = linear_denoiser.medianFilter(13)
    butterworth_denoised = linear_denoiser.butterworthLowPassFilter(20, 4)
    gaussian_filter_denoised = linear_denoiser.gaussianFilter(9, 2)
    bilateral_filter_denoised = linear_denoiser.bilateralFilter(13, 150, 25)
    convolution_2d_denoised = linear_denoiser.Convolution2DFilter(13)
    fft_denoised = linear_denoiser.fftLowPassFilter(55)
    gaussian_low_denoised = linear_denoiser.gaussianLowPassFilter(30)
    wavelet_haar_denoised = linear_denoiser.waveletTransformFilter('haar')
    wavelet_dau_denoised = linear_denoiser.waveletTransformFilter('db1')
    wavelet_bio_denoised = linear_denoiser.waveletTransformFilter('bior1.3')

    box_filter_denoised = cv2.resize(box_filter_denoised, (img_orig.shape[1], img_orig.shape[0]))

    median_filter_denoised = cv2.resize(median_filter_denoised, (img_orig.shape[1], img_orig.shape[0]))
    butterworth_denoised = cv2.resize(butterworth_denoised, (img_orig.shape[1], img_orig.shape[0]))
    gaussian_filter_denoised = cv2.resize(gaussian_filter_denoised, (img_orig.shape[1], img_orig.shape[0]))
    bilateral_filter_denoised = cv2.resize(bilateral_filter_denoised, (img_orig.shape[1], img_orig.shape[0]))
    convolution_2d_denoised = cv2.resize(convolution_2d_denoised, (img_orig.shape[1], img_orig.shape[0]))
    fft_denoised = cv2.resize(fft_denoised, (img_orig.shape[1], img_orig.shape[0]))
    gaussian_low_denoised = cv2.resize(gaussian_low_denoised, (img_orig.shape[1], img_orig.shape[0]))
    wavelet_haar_denoised = cv2.resize(wavelet_haar_denoised, (img_orig.shape[1], img_orig.shape[0]))
    wavelet_dau_denoised = cv2.resize(wavelet_dau_denoised, (img_orig.shape[1], img_orig.shape[0]))
    wavelet_bio_denoised = cv2.resize(wavelet_bio_denoised, (img_orig.shape[1], img_orig.shape[0]))


    linear_psnr_box.append(metrics.psnr(img_orig, box_filter_denoised))
    linear_psnr_median.append(metrics.psnr(img_orig, median_filter_denoised))
    linear_psnr_butterworth.append(metrics.psnr(img_orig, butterworth_denoised))
    linear_psnr_gaussian.append(metrics.psnr(img_orig, gaussian_filter_denoised))
    linear_psnr_bilateral.append(metrics.psnr(img_orig, bilateral_filter_denoised))
    linear_psnr_conv.append(metrics.psnr(img_orig, convolution_2d_denoised))
    linear_psnr_fft.append(metrics.psnr(img_orig, fft_denoised))
    linear_psnr_gauslow.append(metrics.psnr(img_orig, gaussian_low_denoised))
    linear_psnr_wavhaar.append(metrics.psnr(img_orig, wavelet_haar_denoised))
    linear_psnr_wavdau.append(metrics.psnr(img_orig, wavelet_dau_denoised))
    linear_psnr_wavdio.append(metrics.psnr(img_orig, wavelet_bio_denoised))
    psnr_orig.append(metrics.psnr(img_orig, img))



    print("----------------------------------------------------------------------------------------------------------------------\n")
    print("PSNR OF ORIGINAL IMAGE AND NOISY")
    print("Mean error --- PSNR: " + str(np.mean(psnr_orig))+"\n")


    print("----------------------------------------------------------------------------------------------------------------------\n")
    print("LINEAR DENOISING METHODS: EVALUATION \n")

    print("Box Filter --- PSNR: " + str(np.mean(linear_psnr_box)) +"\n")
    print("Median filter --- PSNR: "+ str(np.mean(linear_psnr_median))+"\n")
    print("Butterworth filter --- PSNR: "+str(np.mean(linear_psnr_butterworth))+"\n")
    print("Gaussian filter --- PSNR: "+str(np.mean(linear_psnr_gaussian))+"\n")
    print("Bilateral filter --- PSNR: "+str(np.mean(linear_psnr_bilateral))+"\n")
    print("Convolutional filter --- PSNR: "+str(np.mean(linear_psnr_conv))+"\n")
    print("Fast Fourier Transform --- PSNR: "+str(np.mean(linear_psnr_fft))+"\n")
    print("Gaussian Low Pass filter --- PSNR: "+str(np.mean(linear_psnr_gauslow))+"\n")
    print("Wavelet Transform Haar --- PSNR: "+str(np.mean(linear_psnr_wavhaar))+"\n")
    print("Wavelet Transform Daubechies --- PSNR: "+str(np.mean(linear_psnr_wavdau))+"\n")
    print("Wavelet Transform BiOrthogonal --- PSNR: "+str(np.mean(linear_psnr_wavdio))+"\n")


