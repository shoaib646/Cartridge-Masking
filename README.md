# Cartridge-Masking

## Overview

This repository contains code for an auto-masking system designed specifically for cartridge images. The system uses deep learning techniques to automatically generate masks for cartridge images, facilitating further processing or analysis.

## Getting Started

To use this system, follow these steps:

1. **Clone the Repository**: Clone this repository to your local machine using Git.

    ```bash
    git clone https://github.com/username/Cartridge-Masking.git
    ```

2. **Install Dependencies**: Install the required Python packages by running:

    ```bash
    pip install -r requirements.txt
    ```

3. **Download Data**: Due to limited file size on GitHub, certain data folders (`Stage1_train` and `Stage1_test`) are not included. You can find these folders on Google Drive at [this link](https://drive.google.com/drive/folders/1E5vm-jyJsukBBDQ2-2hMdVGdMDfQyUEb?usp=drive_link). Download them and place them in the appropriate locations within the repository.

4. **Run the Code**:
   - For full model training and steps (estimated time: 1 to 3 minutes), open `ImageSegmentation.ipynb` in Jupyter Notebook and run all cells.
   - To directly check the final output of auto-masking through code, run `Python Modular_File.py`. Make sure to download `model_for_cartridge.h5` and `processed_data.npz` before running the script. Once execution starts, close the performance matrix window to proceed to the end.

## Customization

If you want to adapt this system for your own use case, follow these steps:

1. **Label Training Images**: Use Label-Studio or a similar tool to label your training images.

2. **Organize Image Files**: Organize your labeled images as follows:
   - For training:
     - Place images in the `Stage1_train/image_folder_name/images/` directory.
     - Place corresponding masks in the `Stage1_train/image_folder_name/masks/` directory.
   - For testing:
     - Place images in the `Stage1_test/image_folder/images/` directory.

## Results

To quickly check the results without diving into implementation details, you can:
- Clone the repository or navigate to the `saved_plots` folder to view the results.

## Additional Information

For more details on the implementation or usage of this system, refer to the provided code and documentation within
