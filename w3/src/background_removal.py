import numpy as np
import cv2
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

    def grayscale_conversion(self, image):
        """Convert an image to grayscale manually."""
        return np.dot(image[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)

    def apply_kernel(self, image, kernel):
        """Apply a 3x3 kernel to the image manually using convolution."""
        # Image dimensions
        h, w = image.shape
        # Kernel size
        k_size = kernel.shape[0]

        # Pad the image to handle borders
        pad = k_size // 2
        padded_image = np.pad(image, pad, mode='constant', constant_values=0)

        # Output image
        output = np.zeros((h, w), dtype=np.float32)

        # Convolve kernel over the image
        for i in range(h):
            for j in range(w):
                # Extract the region of interest (ROI)
                region = padded_image[i:i + k_size, j:j + k_size]
                # Perform element-wise multiplication and sum
                output[i, j] = np.sum(region * kernel)
        
        return output

    def compute_gradients_manual(self, image):
        """Compute Sobel gradients manually using convolution."""
        # Convert the image to grayscale
        gray_image = self.grayscale_conversion(image)

        # Define Sobel kernels for x and y gradients
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

        # Apply kernels to the image using manual convolution
        gradient_x = self.apply_kernel(gray_image, sobel_x)
        gradient_y = self.apply_kernel(gray_image, sobel_y)

        # Compute gradient magnitude
        gradient_magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)

        # Normalize the gradient magnitude to range [0, 255]
        gradient_magnitude = np.uint8(255 * gradient_magnitude / np.max(gradient_magnitude))

        return gradient_magnitude

    def detect_contours_from_gradients(self, gradient_magnitude, threshold=80):
        """Detect contours based on gradient magnitude."""
        # Threshold the gradient magnitude to get edges
        edges = np.where(gradient_magnitude > threshold, 255, 0).astype(np.uint8)

        # Find contours using edges
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours by size (e.g., areas that are too small)
        min_area = 5000  # Adjust this threshold as needed to filter out smaller regions
        large_contours = [c for c in contours if cv2.contourArea(c) > min_area]

        return large_contours, edges

    def filter_frame_contours(self, contours, aspect_ratio_range=(0.8, 1.5)):
        """Filter contours that are likely to be picture frames based on area and aspect ratio."""
        filtered_contours = []
        for contour in contours:
            # Get bounding rectangle for each contour
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h
            if aspect_ratio_range[0] <= aspect_ratio <= aspect_ratio_range[1]:
                filtered_contours.append(contour)
        return filtered_contours

    def draw_contours(self, contours):
        """Draw contours on the image for visualization."""
        img_copy = self.image.copy()
        cv2.drawContours(img_copy, contours, -1, (0, 255, 0), 3)  # Green for contours
        self.display_image(img_copy, "Contours Detected")

    def flood_fill_from_contours(self, contours):
        """Flood fill the areas inside the contours."""
        height, width = self.image.shape[:2]
        
        # Create a mask that is 2 pixels larger in both dimensions
        mask = np.zeros((height + 2, width + 2), dtype=np.uint8)
        
        for contour in contours:
            # Get bounding rectangle for the contour
            x, y, w, h = cv2.boundingRect(contour)
            seed_point = (x + w // 2, y + h // 2)  # Seed point in the center of the bounding box
            
            # Use flood fill to fill the area inside the contour
            mask_temp = np.zeros((height + 2, width + 2), dtype=np.uint8)
            cv2.floodFill(self.image, mask_temp, seed_point, 255)
            
            # Combine mask with the current mask
            mask = np.maximum(mask, mask_temp)
        
        # Return the mask, excluding the extra border pixels
        return mask[1:-1, 1:-1]

    def apply_mask(self, mask):
        """Apply the mask to the original image."""
        background_removed = cv2.bitwise_and(self.image, self.image, mask=mask)
        return background_removed

    def morphological_operations_cleanse(self, final_mask):
        """Clean the mask using morphological operations."""
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (100, 1))
        cleaned_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel)
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (100, 100)))
        return cleaned_mask

if __name__ == "__main__":
    # Load the image
    image = cv2.imread('./data/qsd2_w3/00000.jpg')
    background = CalculateBackground(image)

    # Step 1: Compute gradients manually
    gradient_magnitude = background.compute_gradients_manual(image)
    background.display_image(gradient_magnitude, "Manual Gradient Magnitude")

    # Step 2: Detect contours based on manual gradients
    contours, edges = background.detect_contours_from_gradients(gradient_magnitude)

    # Step 3: Draw contours on the image (optional visualization)
    background.draw_contours(contours)

    # Step 4: Apply flood fill using the detected contours
    tot_mask = background.flood_fill_from_contours(contours)

    # Save the flood-filled mask
    cv2.imwrite('./data/qsd2_w2/flood_filled.jpg', cv2.bitwise_not(tot_mask))

    # Apply mask and morphological cleansing
    final_image = background.morphological_operations_cleanse(tot_mask)
    final_image = cv2.bitwise_not(final_image)

    # Save final image
    cv2.imwrite('./data/qsd2_w2/final_image.jpg', final_image)

