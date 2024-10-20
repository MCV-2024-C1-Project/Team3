class LinearDenoiser():
    def __init__(self, image):
        self.img = image

    def boxFilter(self, kernelSize):
        return True
    
    def medianFilter(self, kernelSize):
        return True
    
    def midPointFilter(self, kernelSize):
        return True
    
    def butterworthFilter(self):
        return True
    
    def nyquistFilter(self):
        return True
    
    def gaussianFilter(self):
        return True
    
    def bilateralFilter(self):
        return True
    
    def Convolution2DFilter(self):
        return True
    
    def fftFilter(self):
        return True
    
    def gaussianLowPassFilter(self):
        return True
    
    def waveletTransformFilter(self):
        return True
    

class NonLinearDenoiser():
    def __init__(self, image):
        self.img = image
    
    
    


