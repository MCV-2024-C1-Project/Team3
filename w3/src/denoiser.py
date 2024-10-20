import cv2
import numpy as np
import pywt

class LinearDenoiser():
    def __init__(self, image):
        self.img = image

    def boxFilter(self, kernelSize):
        return cv2.blur(self.img, (kernelSize, kernelSize))
    
    def medianFilter(self, kernelSize):
        return cv2.medianBlur(self.img, kernelSize)
    
    def midPointFilter(self, kernelSize):
        return True
    
    def butterworthLowPassFilter(self, d0, n):

        if len(self.img.shape) == 3:
            # Separate the image color channels
            b, g, r = cv2.split(self.img)

            # Apply the filter to each channel
            b_filtered = self.butterworthGrayLowPass(b, d0, n)
            g_filtered = self.butterworthGrayLowPass(g, d0, n)
            r_filtered = self.butterworthGrayLowPass(r, d0, n)

            # Combine filtered channels
            filtered_image = cv2.merge((b_filtered, g_filtered, r_filtered))
            return filtered_image
        
        else:
            # If the image is on grayscale, apply directly
            return self.butterworthGrayLowPass(self.img, d0, n)
    
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
        return cv2.GaussianBlur(self.img, (kernelSize, kernelSize), sigmaX)
    
    def bilateralFilter(self, d, sigmaColor, sigmaSpace):
        return cv2.bilateralFilter(self.img, d, sigmaColor, sigmaSpace)
    
    def Convolution2DFilter(self, kernelSize):
        kernel = np.ones((kernelSize, kernelSize), np.float32) / (kernelSize * kernelSize)
        return cv2.filter2D(self.img, -1, kernel)
    
    def fftLowPassFilter(self, cutoff):

        if len(self.img.shape) == 3:

            b, g, r = cv2.split(self.img)

            b_filtered = self.fftLowPassGray(b, cutoff)
            g_filtered = self.fftLowPassGray(g, cutoff)
            r_filtered = self.fftLowPassGray(r, cutoff)

            filtered_image = cv2.merge((b_filtered, g_filtered, r_filtered))
            return filtered_image
        else:
            return self.fftLowPassGray(self.img, cutoff)

    
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
        return True
    
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
    
    def waveletTransformFilter(self):

        

        return True
    
    def waveletTransformGray(self, image):

        coeffs = pywt.wavedec2(image, 'haar', level = 2)
        coeffs_thresh = list(map(lambda x: pywt.threshold(x, 10, mode='soft'), coeffs))
        denoised_image = pywt.waverec2(coeffs_thresh, 'haar')
        denoised_image = np.clip(denoised_image, 0, 255)

        return denoised_image.astype(np.uint8)

class NonLinearDenoiser():
    def __init__(self, image):
        self.img = image
    

    


