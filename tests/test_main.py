import os
import unittest
from unittest.mock import patch, MagicMock
from PIL import Image
import torch
from torchvision import transforms, models

from src.main import (
    ImageData,
    ImgProcess,
    Predictor,
)  # replace 'your_module_name' with the name of your module file


class TestImageData(unittest.TestCase):
    @patch("os.listdir")
    @patch("PIL.Image.open")
    def test_load_images(self, mock_open, mock_listdir):
        mock_listdir.return_value = ["image1.jpg", "image2.png", "not_an_image.txt"]
        mock_open.side_effect = [
            MagicMock(spec=Image.Image),
            MagicMock(spec=Image.Image),
        ]

        img_data = ImageData("test_directory")
        images = img_data.load_images()

        self.assertEqual(len(images), 2)
        self.assertTrue(all(isinstance(img, Image.Image) for img in images))


class TestImgProcess(unittest.TestCase):
    @patch("src.main.transforms.Compose")
    def test_resize_and_gray(self, mock_compose):
        mock_transform = MagicMock()
        mock_compose.return_value = mock_transform
        mock_transform.side_effect = lambda x: torch.tensor([1.0])

        img = MagicMock(spec=Image.Image)
        img_list = [img, img]

        img_process = ImgProcess(256)
        processed_images = img_process.resize_and_gray(img_list)

        self.assertEqual(len(processed_images), 2)
        self.assertTrue(all(isinstance(t, torch.Tensor) for t in processed_images))


class TestPredictor(unittest.TestCase):
    @patch("torchvision.models.resnet18")
    def test_predict_img(self, mock_resnet18):
        mock_model = MagicMock()
        mock_resnet18.return_value = mock_model
        mock_model.eval.return_value = None
        mock_model.side_effect = lambda x: torch.tensor([[0.1, 0.9]])

        predictor = Predictor()
        img_tensor = torch.tensor([1.0, 2.0, 3.0])
        processed_images = [img_tensor, img_tensor]

        results = predictor.predict_img(processed_images)

        self.assertEqual(len(results), 2)
        self.assertTrue(
            all(isinstance(r, int) or isinstance(r, float) for r in results)
        )


if __name__ == "__main__":
    unittest.main()
