mport cv2
import os
import argparse
from albumentations import (
    HorizontalFlip, RandomBrightnessContrast, Rotate, RandomCrop, Resize, 
    GaussianBlur, GaussNoise, CLAHE, HueSaturationValue, Compose)
from albumentations.augmentations.transforms import MotionBlur
class ImageAugmentor:
    def _init_(self, output_dir):
        """
        Initialize the augmentor with the directory to store augmented images.
        Args:
            output_dir (str): Directory to save augmented images.
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.augmentation_pipeline = Compose([
            HorizontalFlip(p=0.5),
            RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            Rotate(limit=15, p=0.5),
            RandomCrop(height=300, width=300, p=0.5),
            Resize(height=512, width=512, p=1.0),
            GaussianBlur(blur_limit=(3, 7), p=0.3),
            GaussNoise(var_limit=(10.0, 50.0), p=0.4),
            CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.4),
            HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
            MotionBlur(blur_limit=(3, 7), p=0.3)
        ])
    def augment_and_save(self, image_path):
        """
        Apply augmentation to a single image and save the result.
        Args:
            image_path (str): Path to the input image.
        """
        image_name = os.path.basename(image_path).split('.')[0]
        image = cv2.imread(image_path)
        if image is not None:
            augmented = self.augmentation_pipeline(image=image)
            augmented_image = augmented["image"]
            output_path = os.path.join(self.output_dir, f"{image_name}_augmented.jpg")
            cv2.imwrite(output_path, augmented_image)
            print(f"Saved augmented image: {output_path}")
        else:
            print(f"Error reading image: {image_path}")
    def augment_directory(self, input_dir):
        """
        Apply augmentation to all images in the given directory.
        Args:
            input_dir (str): Directory containing input images.
        """
        for image_name in os.listdir(input_dir):
            if image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                self.augment_and_save(os.path.join(input_dir, image_name))
if _name_ == "_main_":
    parser = argparse.ArgumentParser(description="Augment images in a directory")
    parser.add_argument("--input", type=str, required=True, help="Directory containing input images")
    parser.add_argument("--output", type=str, required=True, help="Directory to save augmented images")
    args = parser.parse_args()
    augmentor = ImageAugmentor(args.output)
    augmentor.augment_directory(args.input)
