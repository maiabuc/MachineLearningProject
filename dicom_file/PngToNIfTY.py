from PIL import Image
import numpy as np
import os
import nibabel as nib

def convert_images_to_nifti(png_dir, output_dir):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # List all PNG files in the input directory
    png_files = [f for f in os.listdir(png_dir) if f.endswith('.png')]

    for png_file in png_files:
        # Open the PNG image
        img = Image.open(os.path.join(png_dir, png_file))

        # Convert image to numpy array
        img_array = np.array(img)

        # Create a NIfTI image
        nifti_img = nib.Nifti1Image(img_array, np.eye(4))

        # Save the NIfTI image to the output directory with the same filename
        nib.save(nifti_img, os.path.join(output_dir, png_file.replace('.png', '.nii')))
        print(f"Converted {png_file} to {png_file.replace('.png', '.nii')}")


# Example usage

convert_images_to_nifti('./TestLabelsP', './TestLabels')
convert_images_to_nifti('./TrainImagesP', './TrainImages')
convert_images_to_nifti('./TrainLabelsP', './TrainLabels')
