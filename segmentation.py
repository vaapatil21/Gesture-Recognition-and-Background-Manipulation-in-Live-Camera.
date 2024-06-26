import torch
import numpy as np
import cv2 as cv
from torchvision import models
import torchvision.transforms as T

class HumanSegmenter:
    def __init__(self, alpha=0.5):
        """
        Initialize the HumanSegmenter with a pre-trained DeepLabV3 model, transformation functions, and blending alpha.
        :param alpha (float): Blending factor for combining the original image and the colorized segmentation mask.
        """
        self.model = models.segmentation.deeplabv3_resnet101(pretrained=True).eval()
        self.transforms = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.alpha = alpha

    def segment_human(self, img):
        """
        Perform human segmentation on the input image using the pre-trained DeepLabV3 model.
        :param img (np.ndarray): Input image.
        :return segmentation_mask (np.ndarray): Segmentation mask where human pixels are marked with class 15.
        """
        input_tensor = self.transforms(img)
        input_batch = input_tensor.unsqueeze(0)
        with torch.no_grad():
            output = self.model(input_batch)['out'][0]
        output_predictions = output.argmax(0)
        return output_predictions.numpy()

    def colorize_segmentation(self, segmentation_mask: np.ndarray, img: np.ndarray, background_color) -> np.ndarray:
        """
        Colorize the segmentation mask based on the specified background color.
        :param segmentation_mask (np.ndarray): Segmentation mask.
        :param img (np.ndarray): Original image.
        :param background_color (int): Background color selection (0 for green, 1 for blue).
        :returns: colorized_mask (np.ndarray): Colorized image based on the segmentation mask and background color.
        """
        if background_color == 0:
            background_color = (0, 0, 255)

        else: #background_color == 1:
            background_color = (0, 255, 0)


        colorized_mask = img.copy()
        background = np.zeros_like(img)
        background[:] = background_color

        # Set non-human regions to the specified background color
        colorized_mask[segmentation_mask != 15] = background[segmentation_mask != 15]

        return colorized_mask

    def detect_and_draw(self, img: np.ndarray , background_color) -> np.ndarray:
        """
        Detect and draw the segmented human in the original image.
        :param img (np.ndarray): Original image.
        :param background_color (int): Background color selection (0 for green, 1 for blue).
        :returns: blended_image (np.ndarray): Blended image with the original and colorized segments.
        """
        if background_color == -1:
            return img
        segmentation_mask = self.segment_human(img)
        colorized_mask = self.colorize_segmentation(segmentation_mask, img, background_color)
        blended_image = cv.addWeighted(img, 1 - self.alpha, colorized_mask, self.alpha, 0)

        return blended_image

if __name__ == "__main__":
    
    # Create an instance of the HumanSegmenter
    segmenter = HumanSegmenter()

    # Open a video capture object
    cap = cv.VideoCapture(0)  # Use 0 for the default camera, change it if you have multiple cameras

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Perform segmentation and display the result
        background_color = int(input("Enter 0 for green and 1 for blue"))
        result = segmenter.detect_and_draw(frame,background_color)

        cv.imshow("Segmented Image", result)

        # Break the loop if 'q' key is pressed
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture object and close all windows
    cap.release()
    cv.destroyAllWindows()
