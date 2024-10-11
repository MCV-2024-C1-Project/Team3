import cv2
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
import os

class CalculateBackground():
    def __init__(self, image):
        self.image = image

    def display_image(self, img, title):
        """Utility function to display images."""
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(title)
        plt.axis('off')
        plt.show()
    
    def color_thresholding_simple(self, threshold, image = [], mask_des = []):

        if len(image) == 0:
            image = self.image

        lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
        l_channel = lab_image[:, :, 0]
        a_channel = lab_image[:, :, 1]
        b_channel = lab_image[:, :, 2]

        lower_bound = np.array([0, -15, -15])  # Valor L muy bajo para incluir sombras muy oscuras
        upper_bound = np.array([60, 15, 15])    # Valor L un poco más alto para grises oscuros

        mask = cv2.inRange(lab_image, lower_bound, upper_bound)

        if len(mask_des) > 0:
            mask = mask + mask_des

        return mask

    def convert_to_lab(self, image):
        """Convert the image to CIELAB color space."""
        return cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    def create_mask(self, shape):
        """Create an empty mask for the region."""
        return np.zeros(shape[:2], dtype=np.uint8)
    
    def get_seed_color(self, lab_image, seed_point):
        """Get the color of the seed point."""
        return lab_image[seed_point[1], seed_point[0]]  # (x, y)

    def adaptive_thresholding(self,image):
        """Apply adaptive thresholding to the image."""
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    def flood_fill_region_with_edges(self, seed_point, tolerance=20, edge_map=None):
        """Perform region growing using OpenCV's floodFill and stop at edges."""
        lab_image = self.convert_to_lab(self.image)
        height, width = lab_image.shape[:2]

        # Expand mask size to include a 1-pixel border
        mask = self.create_mask((height + 2, width + 2))

        # Incorporar edge_map en la máscara
        if edge_map is not None:
            mask[1:height+1, 1:width+1] = np.where(edge_map == 0, 1, 0)

        seed_color = self.get_seed_color(lab_image, seed_point)
        lower_bound = (tolerance, tolerance, tolerance)
        upper_bound = (tolerance, tolerance, tolerance)

        # Flood fill parameters
        flags = 4 | (255 << 8)  # 4-connectivity and a fixed value of 255

        # Apply flood fill
        cv2.floodFill(lab_image, mask, seed_point, 255, lower_bound, upper_bound, flags)

        # Remove the border added earlier
        final_mask = mask[1:-1, 1:-1]



        return final_mask




    def apply_mask(self, mask):
        """Apply the mask to the original image to extract the foreground."""
        # Invert the mask to consider everything outside the region
        #inverted_mask = cv2.bitwise_not(mask)
        
        # Apply the inverted mask to get the background
        background_removed = cv2.bitwise_and(self.image, self.image, mask=mask)
        
        return background_removed
    

    
    def morphological_operations_cleanse(self, final_mask):

        # Morphological operations to eliminate noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))

        # Apply opening to remove small noise (erosion followed by dilation)
        #cleaned_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel)

        # Optionally, apply closing to fill small holes (dilation followed by erosion)
        cleaned_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1)))

        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_DILATE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        
        return cleaned_mask
    

if __name__ == "__main__":

    iou_scores = []

    for image_name in os.listdir("./data/qsd2_w2"):

        if image_name.endswith(".jpg"):

            image = cv2.imread("./data/qsd2_w2/"+image_name)
            background = CalculateBackground(image)


            seed_points = [
                (0, 0),  # Top-left corner
                (image.shape[1] - 1, 0),  # Top-right corner
                (0, image.shape[0] - 1),  # Bottom-left corner
                (image.shape[1] - 1, image.shape[0] - 1),  # Bottom-right corner
            ]

            edge_map = background.adaptive_thresholding(image)

            #background.display_image(edge_map, "Adaptive Gaussian Thresholding mask")

        # Perform region growing for each seed point and stack the masks
            tot_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            for seed in seed_points:
                mask = background.flood_fill_region_with_edges(seed, tolerance=5, edge_map=edge_map)
                tot_mask = np.maximum(tot_mask, mask)

            # Apply the mask to get the foreground
            foreground = background.apply_mask(tot_mask)

            final_mask = background.color_thresholding_simple(0, foreground)

            tot_mask = tot_mask + final_mask

            final_image = background.morphological_operations_cleanse(tot_mask)
            final_image = cv2.bitwise_not(final_image)

            cv2.imwrite("./data/results/"+image_name, final_image)

            # Load ground truth
            gt = cv2.imread("./data/qsd2_w2/"+image_name[:-4]+".png", cv2.IMREAD_GRAYSCALE)

            # Calculate IoU
            intersection = np.logical_and(gt, final_image)
            union = np.logical_or(gt, final_image)
            iou_score = np.sum(intersection) / np.sum(union)

            iou_scores.append(iou_score)

    print(f"Mean IoU: {np.mean(iou_scores)}")