import os
from PIL import Image
import torch  # type: ignore
from torchvision import transforms, models  # type: ignore
from torchvision.models.resnet import ResNet18_Weights  # type: ignore


class ImageData:
    """
    A class to load images from a specified directory.

    Attributes:
    -----------
    D : str
        The directory path where images are stored.

    Methods:
    --------
    load_images() -> list[Image.Image]:
        Loads all .jpg and .png images from the directory and returns them as a list of PIL Images.
    """

    def __init__(self, DIR: str):
        """
        Initializes the ImageData with the directory path.

        Parameters:
        -----------
        DIR : str
            The directory path where images are stored.
        """
        self.D = DIR

    def load_images(self) -> list[Image.Image]:
        """
        Loads all .jpg and .png images from the directory.

        Returns:
        --------
        list[Image.Image]
            A list of loaded PIL Image objects.
        """
        imgs = []
        for F in os.listdir(self.D):
            if F.endswith(".jpg") or F.endswith(".png"):
                imgs.append(Image.open(os.path.join(self.D, F)))
        return imgs


class ImgProcess:
    """
    A class to process images by resizing and converting them to grayscale.

    Attributes:
    -----------
    s : int
        The size to which images will be resized (both width and height).

    Methods:
    --------
    resize_and_gray(img_list: list[Image.Image]) -> list[torch.Tensor]:
        Resizes and converts images to grayscale, then transforms them into normalized tensors.
    """

    def __init__(self, size: int):
        """
        Initializes the ImgProcess with the target size.

        Parameters:
        -----------
        size : int
            The size to which images will be resized (both width and height).
        """
        self.s = size

    def resize_and_gray(self, img_list: list[Image.Image]) -> list[torch.Tensor]:
        """
        Resizes images to the specified size, converts them to grayscale, and normalizes them.

        Parameters:
        -----------
        img_list : list[Image.Image]
            A list of PIL Image objects to be processed.

        Returns:
        --------
        list[torch.Tensor]
            A list of processed image tensors.
        """
        p_images = []
        for img in img_list:
            t = transforms.Compose(
                [
                    transforms.Resize((self.s, self.s)),
                    transforms.Grayscale(num_output_channels=3),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
            p_images.append(t(img))
        return p_images


class Predictor:
    """
    A class to predict the class of processed images using a pre-trained ResNet18 model.

    Attributes:
    -----------
    mdl : torch.nn.Module
        The pre-trained ResNet18 model.

    Methods:
    --------
    predict_img(processed_images: list[torch.Tensor]) -> list[int | float]:
        Predicts the class for each processed image tensor.
    """

    def __init__(self):
        """
        Initializes the Predictor with a pre-trained ResNet18 model.
        """
        self.mdl = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.mdl.eval()

    def predict_img(self, processed_images: list[torch.Tensor]) -> list[int | float]:
        """
        Predicts the class for each processed image tensor.

        Parameters:
        -----------
        processed_images : list[torch.Tensor]
            A list of processed image tensors.

        Returns:
        --------
        list[int | float]
            A list of predicted classes for each image tensor.
        """
        results = []
        for img_tensor in processed_images:
            pred = self.mdl(img_tensor.unsqueeze(0))
            results.append(torch.argmax(pred, dim=1).item())
        return results


if __name__ == "__main__":
    loader = ImageData("images/")
    images = loader.load_images()

    processor = ImgProcess(256)
    processed_images = processor.resize_and_gray(images)

    pred = Predictor()
    results = pred.predict_img(processed_images)
    print(results)
