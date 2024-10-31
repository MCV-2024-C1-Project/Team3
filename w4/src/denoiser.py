import cv2
import numpy as np
import pywt
import os
from math import log10, sqrt
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
import bm3d

class LinearDenoiser():
    def __init__(self, image):
        self.img = image

    def boxFilter(self, kernelSize):
        img=self.img.copy()
        return cv2.blur(img, (kernelSize, kernelSize))
    
    def medianFilter(self, kernelSize):
        img = self.img.copy()
        return cv2.medianBlur(img, kernelSize)
    
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

    def rof_denoising(self, lambda_val = 0.2, n_iter = 300, tau = 0.15, tol = 1e-6):
        img = self.img.copy()
        if len(img.shape) == 3:
            yuv_image = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
            y_channel = yuv_image[:, :, 0]
            y_denoised = self.rof_denoise_channel(y_channel,lambda_val, n_iter, tau, tol)
            yuv_image[:, :, 0] = y_denoised
            denoised_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR)
            return denoised_image
        else:
            return self.rof_denoise_channel(img, lambda_val, n_iter, tau, tol)

    def rof_denoise_channel(self, channel_img, lambda_val, n_iter, tau, tol):
        # Initial values
        u = channel_img.astype(np.float64)
        px = np.zeros_like(channel_img)
        py = np.zeros_like(channel_img)
        grad_norm = np.zeros_like(channel_img)

        # Gradient operators (forward difference)
        grad_x = lambda u: np.roll(u, -1, axis=1) - u
        grad_y = lambda u: np.roll(u, -1, axis=0) - u

        # Divergence operator (backward difference)
        div = lambda px, py: np.roll(px, 1, axis=1) - px + np.roll(py, 1, axis=0) - py

        for i in tqdm(range(n_iter)):
            # Compute gradient of u
            u_grad_x = grad_x(u)
            u_grad_y = grad_y(u)

            # Update dual variables
            px_new = px + tau * u_grad_x
            py_new = py + tau * u_grad_y

            # Compute the norm of the dual variables
            grad_norm = np.maximum(1.0, np.sqrt(px_new**2 + py_new**2))

            # Normalize the dual variables
            px = px_new / grad_norm
            py = py_new / grad_norm

            # Update the denoised image u
            u_old = u
            u = channel_img + lambda_val * div(px, py)

            # Check for convergence
            if np.linalg.norm(u - u_old) / np.linalg.norm(u) < tol:
                break

        return u

    def non_local_means_denoising(self, h = 20, templateWindowSize = 20, searchWindowSize = 10):
        img = self.img.copy()
        # Ensure image is in 8-bit format
        if img.dtype != np.uint8:
            img_8u = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        else:
            img_8u = img


        # If the image is color (3 channels), use the color version of the function
        if len(img_8u.shape) == 3:
            denoised_image = cv2.fastNlMeansDenoisingColored(img_8u, None, h, templateWindowSize, searchWindowSize)
        else:
            # For grayscale images, use the normal version
            denoised_image = cv2.fastNlMeansDenoising(img_8u, None, h, templateWindowSize, searchWindowSize)

        # Convert back to uint8 format if the original image was in that type
        return np.uint8(np.clip(denoised_image * 255, 0, 255))

    def bmd3_denoising_byhand(self, block_size = 9, search_window = 21, h = 10):
        img = self.img.copy()

        padded_img = cv2.copyMakeBorder(img, block_size // 2, block_size // 2,
                                            block_size // 2, block_size // 2,
                                            cv2.BORDER_REFLECT)
        denoised_img = np.zeros_like(img)

        for i in tqdm(range(img.shape[0])):
            for j in range(img.shape[1]):
                # Extract the current block
                current_block = padded_img[i:i + block_size, j: j + block_size]

                # Initialize weight and pixel accumulator
                weight_sum = 0
                denoised_pixel = 0

                # Search for similar blocks
                for m in range(-search_window // 2, search_window // 2 + 1):
                    for n in range(-search_window // 2, search_window // 2 + 1):
                        # Get coordinates for similar block
                        y = i + block_size // 2 + m
                        x = j + block_size // 2 + n

                        # Ensure we are within the image bounds
                        if y >= 0 and y + block_size <= padded_img.shape[0] and x >= 0 and x + block_size <= padded_img.shape[1]:
                            similar_block = padded_img[y:y + block_size, x:x + block_size]

                            # Calculate the weight based on similarity
                            distance = np.sum((current_block - similar_block) ** 2)
                            weight = np.exp(-distance / (h ** 2))

                            # Accumulate weighted pixel values
                            denoised_pixel += weight * padded_img[y + block_size // 2, x + block_size // 2]
                            weight_sum += weight
                
                # Final denoised pixel value
                if weight_sum > 0:
                    denoised_img[i, j] = denoised_pixel / weight_sum
                else:
                    denoised_img[i, j] = img[i, j]

        return np.clip(denoised_img, 0, 255). astype(np.uint8)
    
    def bm3d_denoising(self, sigma_psd = 25, stage_arg = bm3d.BM3DStages.ALL_STAGES):
        img = self.img.copy()

        if len(img.shape) == 3:
            b, g, r = cv2.split(img)
            b_denoised = bm3d.bm3d(b, sigma_psd=25, stage_arg=stage_arg)
            g_denoised = bm3d.bm3d(g, sigma_psd=25, stage_arg=stage_arg)
            r_denoised = bm3d.bm3d(r, sigma_psd=25, stage_arg=stage_arg)
            return cv2.merge((b_denoised, g_denoised, r_denoised))
        else:
            return bm3d.bm3d(img, sigma_psd=sigma_psd, stage_arg=stage_arg)


    
    def waveletShrinkage3D(self, img, wavelet = 'bior1.3', level = 3, threshold_factor = 0.75):


        if len(img.shape) == 3:
            b, g, r = cv2.split(img)
            b_denoised = self.waveletShrinkage(b, wavelet, level, threshold_factor)
            g_denoised = self.waveletShrinkage(g, wavelet, level, threshold_factor)
            r_denoised = self.waveletShrinkage(r, wavelet, level, threshold_factor)
            return cv2.merge((b_denoised, g_denoised, r_denoised))
        else:
            return self.waveletShrinkage(img, wavelet, level, threshold_factor)
        
    
    def adaptive_threshold(self, data, sigma, level, max_level, factor):
        """
        Applies adaptive thresholding based on the detail level.
        The threshold decreases with increasing level of detail.
        """
        # Scale the threshold by detail level to provide adaptive thresholding
        level_factor = factor * (2 ** ((max_level - level) / 2))
        threshold = level_factor * sigma * np.sqrt(2 * np.log2(data.size))
        return np.sign(data) * np.maximum(np.abs(data) - threshold, 0)

    def waveletShrinkage(self, img, wavelet='bior1.3', level=3, threshold_factor=3):
        
        # Perform wavelet decomposition
        coeffs = pywt.wavedec2(img, wavelet, level)
        
        # Estimate the noise standard deviation from the first level coefficients
        sigma_est = np.median(np.abs(coeffs[-1])) / 0.6745

        # Apply adaptive thresholding to the detail coefficients
        thresholded_coeffs = [coeffs[0]]
        for i, detail_level in enumerate(coeffs[1:], 1):
            thresholded_level = tuple(
                self.adaptive_threshold(data, sigma_est, i, level, threshold_factor) for data in detail_level
            )
            thresholded_coeffs.append(thresholded_level)
        
        # Reconstruct the image using the thresholded coefficients
        denoised_image = pywt.waverec2(thresholded_coeffs, wavelet)

        # Clip values and rescale back to [0, 255]
        return np.clip(denoised_image, 0, 255).astype(np.uint8)

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
    
def denoiseOne(image_path, name):
    metrics = NoiseMetric()

    img = cv2.imread(image_path+name)

    denoiser = NonLinearDenoiser(img)

    wavelet_denoised = denoiser.waveletShrinkage3D(img, wavelet='bior1.3', level=3, threshold_factor=0.75)
    wav_median_denoised = denoiser.waveletShrinkage3D(cv2.medianBlur(img, 5), wavelet='bior1.3', level=3, threshold_factor=0.75)

    wavelet_denoised = cv2.resize(wavelet_denoised, (img.shape[1], img.shape[0]))
    wav_median_denoised = cv2.resize(wav_median_denoised, (img.shape[1], img.shape[0]))

    wavelet_psnr = metrics.psnr(img, wavelet_denoised)
    wav_median_psnr = metrics.psnr(img, wav_median_denoised)

    if wavelet_psnr > wav_median_psnr:
        return wavelet_denoised
    else:
        return wav_median_denoised
    
def denoiseAll(image_path, denoised_path):

    metrics = NoiseMetric()

    for name in tqdm(os.listdir(image_path)):
        if name.endswith('.jpg'):
            img = cv2.imread(os.path.join(image_path, name))
            denoiser = NonLinearDenoiser(img)
            
            wavelet_denoised = denoiser.waveletShrinkage3D(img, wavelet='bior1.3', level=3, threshold_factor=0.75)
            wav_median_denoised = denoiser.waveletShrinkage3D(cv2.medianBlur(img, 5), wavelet='bior1.3', level=3, threshold_factor=0.75)

            wavelet_denoised = cv2.resize(wavelet_denoised, (img.shape[1], img.shape[0]))
            wav_median_denoised = cv2.resize(wav_median_denoised, (img.shape[1], img.shape[0]))

            wavelet_psnr = metrics.psnr(img, wavelet_denoised)
            wav_median_psnr = metrics.psnr(img, wav_median_denoised)

            if wavelet_psnr > wav_median_psnr:
                cv2.imwrite(os.path.join(denoised_path, name), wavelet_denoised)

            else:
                cv2.imwrite(os.path.join(denoised_path, name), wav_median_denoised)
    
def test():

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
    non_linear_psnr_nlm = []
    non_linear_psnr_bmd3 = []
    non_linear_psnr_bm3d_hand = []
    non_linear_psnr_wavelet = []
    non_linear_bilateral_median_psnr = []
    wav_bilateral_psnr = []
    wav_median_psnr = []

    info_images = [0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    info_dif = []


    for image_path in tqdm(os.listdir('./data/qsd1_w3')):
        if image_path.endswith('.jpg'):

            img = cv2.imread('./data/qsd1_w3/'+image_path)
            img_orig = cv2.imread('./data/non_augmented/'+image_path)


            linear_denoiser = LinearDenoiser(img)
            non_linear_denoiser = NonLinearDenoiser(img)

            box_filter_denoised = linear_denoiser.boxFilter(11)
            median_filter_denoised = linear_denoiser.medianFilter(11)
            butterworth_denoised = linear_denoiser.butterworthLowPassFilter(30, 2)
            gaussian_filter_denoised = linear_denoiser.gaussianFilter(9, 2)
            bilateral_filter_denoised = linear_denoiser.bilateralFilter(13, 100, 75)
            convolution_2d_denoised = linear_denoiser.Convolution2DFilter(11)
            fft_denoised = linear_denoiser.fftLowPassFilter(20)
            gaussian_low_denoised = linear_denoiser.gaussianLowPassFilter(30)
            wavelet_haar_denoised = linear_denoiser.waveletTransformFilter('haar')
            wavelet_dau_denoised = linear_denoiser.waveletTransformFilter('db1')
            wavelet_bio_denoised = linear_denoiser.waveletTransformFilter('bior1.3')
            nlm_filter_denoised = non_linear_denoiser.non_local_means_denoising()
            bm3d_filter_denoised = non_linear_denoiser.bm3d_denoising()
            wav_filter_denoised = non_linear_denoiser.waveletShrinkage3D(img)
            bilat_median_denoised = cv2.medianBlur(cv2.bilateralFilter(img, 9, 75, 75), 5)
            wav_bilateral_denoised = non_linear_denoiser.waveletShrinkage3D(cv2.bilateralFilter(img, 9, 75, 75), wavelet='bior1.3', level=3, threshold_factor=0.75)
            #wav_bilateral_denoised = cv2.bilateralFilter(non_linear_denoiser.waveletShrinkage3D(wavelet='bior1.3', level=3, threshold_factor=3), 9, 75, 75)
            wav_median_denoised = non_linear_denoiser.waveletShrinkage3D(cv2.medianBlur(img, 5), wavelet='bior1.3', level=3, threshold_factor=0.75)
            #wav_median_denoised = cv2.medianBlur(non_linear_denoiser.waveletShrinkage3D(wavelet='bior1.3', level=3, threshold_factor=3), 11)



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
            nlm_filter_denoised = cv2.bitwise_not(cv2.resize(nlm_filter_denoised, (img_orig.shape[1], img_orig.shape[0])))
            bm3d_filter_denoised = cv2.resize(bm3d_filter_denoised, (img_orig.shape[1], img_orig.shape[0]))
            wav_filter_denoised = cv2.resize(wav_filter_denoised, (img_orig.shape[1], img_orig.shape[0]))
            bilat_median_denoised = cv2.resize(bilat_median_denoised, (img_orig.shape[1], img_orig.shape[0]))
            wav_bilateral_denoised = cv2.resize(wav_bilateral_denoised, (img_orig.shape[1], img_orig.shape[0]))
            wav_median_denoised = cv2.resize(wav_median_denoised, (img_orig.shape[1], img_orig.shape[0]))

            cv2.imwrite('./data/denoised_images/box_filter_'+image_path, box_filter_denoised)
            cv2.imwrite('./data/denoised_images/median_filter_'+image_path, median_filter_denoised)
            cv2.imwrite('./data/denoised_images/butterworth_filter_'+image_path, butterworth_denoised)
            cv2.imwrite('./data/denoised_images/gaussian_filter_'+image_path, gaussian_filter_denoised)
            cv2.imwrite('./data/denoised_images/bilateral_filter_'+image_path, bilateral_filter_denoised)
            cv2.imwrite('./data/denoised_images/convolution_2d_filter_'+image_path, convolution_2d_denoised)
            cv2.imwrite('./data/denoised_images/fft_filter_'+image_path, fft_denoised)
            cv2.imwrite('./data/denoised_images/gaussian_low_filter_'+image_path, gaussian_low_denoised)
            cv2.imwrite('./data/denoised_images/wavelet_haar_filter_'+image_path, wavelet_haar_denoised)
            cv2.imwrite('./data/denoised_images/wavelet_dau_filter_'+image_path, wavelet_dau_denoised)
            cv2.imwrite('./data/denoised_images/wavelet_bio_filter_'+image_path, wavelet_bio_denoised)
            cv2.imwrite('./data/denoised_images/nlm_filter_'+image_path, nlm_filter_denoised)
            cv2.imwrite('./data/denoised_images/bm3d_filter_'+image_path, bm3d_filter_denoised)
            cv2.imwrite('./data/denoised_images/wav_filter_'+image_path, wav_filter_denoised)
            cv2.imwrite('./data/denoised_images/bilat_median_filter_'+image_path, bilat_median_denoised)
            cv2.imwrite('./data/denoised_images/wav_bilateral_filter_'+image_path, wav_bilateral_denoised)
            cv2.imwrite('./data/denoised_images/wav_median_filter_'+image_path, wav_median_denoised)



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
            non_linear_psnr_nlm.append(metrics.psnr(img_orig, nlm_filter_denoised))
            non_linear_psnr_bmd3.append(metrics.psnr(img_orig, bm3d_filter_denoised))
            non_linear_psnr_wavelet.append(metrics.psnr(img_orig, wav_filter_denoised))
            non_linear_bilateral_median_psnr.append(metrics.psnr(img_orig, bilat_median_denoised))
            wav_bilateral_psnr.append(metrics.psnr(img_orig, wav_bilateral_denoised))
            wav_median_psnr.append(metrics.psnr(img_orig, wav_median_denoised))
            info_dif.append({
                'name': image_path,
                'psnr': metrics.psnr(img_orig, img),
                'psnr_denoised':{
                    'box_filter': metrics.psnr(img_orig, box_filter_denoised),
                    'median_filter': metrics.psnr(img_orig, median_filter_denoised),
                    'butterworth_filter': metrics.psnr(img_orig, butterworth_denoised),
                    'gaussian_filter': metrics.psnr(img_orig, gaussian_filter_denoised),
                    'bilateral_filter': metrics.psnr(img_orig, bilateral_filter_denoised),
                    'convolution_2d_filter': metrics.psnr(img_orig, convolution_2d_denoised),
                    'fft_filter': metrics.psnr(img_orig, fft_denoised),
                    'gaussian_low_filter': metrics.psnr(img_orig, gaussian_low_denoised),
                    'wavelet_haar_filter': metrics.psnr(img_orig, wavelet_haar_denoised),
                    'wavelet_dau_filter': metrics.psnr(img_orig, wavelet_dau_denoised),
                    'wavelet_bio_filter': metrics.psnr(img_orig, wavelet_bio_denoised),
                    'nlm_filter': metrics.psnr(img_orig, nlm_filter_denoised),
                    'bm3d_filter': metrics.psnr(img_orig, bm3d_filter_denoised),
                    'wav_filter': metrics.psnr(img_orig, wav_filter_denoised),
                    'non_linear_bilateral_median_psnr': metrics.psnr(img_orig, bilat_median_denoised),
                    'wav_bilateral_psnr': metrics.psnr(img_orig, wav_bilateral_denoised),
                    'wav_median_psnr': metrics.psnr(img_orig, wav_median_denoised)
                }
                
                    
            })





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

    print("----------------------------------------------------------------------------------------------------------------------\n")
    print("NON LINEAR DENOISING METHODS: EVALUATION \n")

    print("NLM Filter --- PSNR: "+str(np.mean(non_linear_psnr_nlm))+"\n")
    print("BM3D Filter --- PSNR: "+str(np.mean(non_linear_psnr_bmd3))+"\n")
    print("Wavelet Shrinkage --- PSNR: "+str(np.mean(non_linear_psnr_wavelet))+"\n")
    print("Bilateral Median --- PSNR: "+str(np.mean(non_linear_bilateral_median_psnr))+"\n")
    print("Wavelet Bilateral --- PSNR: "+str(np.mean(wav_bilateral_psnr))+"\n")
    print("Wavelet Median --- PSNR: "+str(np.mean(wav_median_psnr))+"\n")

    print("----------------------------------------------------------------------------------------------------------------------\n")
    print("INFO IMAGES: \n")
    print(info_dif)

if __name__ == '__main__':
    test()



