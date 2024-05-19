import os
from glob import glob
from monai.transforms import (
    Compose,
    LoadImage,
    AddChanneld,
    Resized,
    ToTensord,
)
from monai.data import DataLoader, Dataset, CacheDataset
from monai.utils import set_determinism

def prepare(in_dir, spatial_size=(128, 128), cache=True):
    """
    This function is for preprocessing, it contains only the basic transforms, but you can add more operations that you
    find in the Monai documentation.
    https://monai.io/docs.html
    """

    set_determinism(seed=0)

    # Define paths to training and test images and labels
    path_train_images = sorted(glob(os.path.join(in_dir, "TrainImages", "*.png")))
    path_train_labels = sorted(glob(os.path.join(in_dir, "TrainLabels", "*.png")))

    path_test_images = sorted(glob(os.path.join(in_dir, "TestImages", "*.png")))
    path_test_labels = sorted(glob(os.path.join(in_dir, "TestLabels", "*.png")))

    # Create dictionaries of file paths for training and testing
    train_files = [{"image": image_name, "label": label_name} for image_name, label_name in
                   zip(path_train_images, path_train_labels)]
    test_files = [{"image": image_name, "label": label_name} for image_name, label_name in
                  zip(path_test_images, path_test_labels)]

    # Define transformations for training and test data
    train_transforms = Compose(
        [
            LoadImage(image_only=True),
            AddChanneld(),
            Resized(spatial_size=spatial_size),
            ToTensord(),
        ]
    )

    test_transforms = Compose(
        [
            LoadImage(image_only=True),
            AddChanneld(),
            Resized(spatial_size=spatial_size),
            ToTensord(),
        ]
    )

    # Create DataLoader objects
    if cache:
        train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0)
        train_loader = DataLoader(train_ds, batch_size=1)

        test_ds = CacheDataset(data=test_files, transform=test_transforms, cache_rate=1.0)
        test_loader = DataLoader(test_ds, batch_size=1)
    else:
        train_ds = Dataset(data=train_files, transform=train_transforms)
        train_loader = DataLoader(train_ds, batch_size=1)

        test_ds = Dataset(data=test_files, transform=test_transforms)
        test_loader = DataLoader(test_ds, batch_size=1)

    return train_loader, test_loader

