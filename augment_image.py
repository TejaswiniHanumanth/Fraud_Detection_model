from torchvision import transforms
from PIL import Image

def augment_image(image_path, output_dir):
    """
    Apply data augmentation to an image and save the results.
    Args:
        image_path (str): Path to the input image.
        output_dir (str): Directory to save augmented images.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    img = Image.open(image_path)
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0))
    ])
    
    for i in range(5):  # Generate 5 augmented versions
        augmented_img = transform(img)
        augmented_img.save(os.path.join(output_dir, f"aug_{i}_{os.path.basename(image_path)}"))

# Example usage:
augment_image("frame_00001.jpg", "augmented_frames")
