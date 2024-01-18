# Cartridge-Masking
## Step: Clone this directory
## Best Suggestion (saved model high accuracy): clone directory --> install requirements.txt --> Run Python Moduler_File.py

## List of Folders that you won't find here (Due to limited file size on Github):
    - Stage1_train & Stage1_test, Find those @ (Google Drive)[https://drive.google.com/drive/folders/1E5vm-jyJsukBBDQ2-2hMdVGdMDfQyUEb?usp=drive_link]

## Execution
    - If you want to check whole model training  & Steps (Estimated Time: 1 to 3 minutes) (Jupyter-Notebook).
        * Open ImageSegmentation.ipynb --> Run All Cells.

    - If you want to just check with final output of Auto Masking through code.
        * Run --> Python Modular_File.py (Make sure to download model_for_cartridge.h5 & processed_data.npz) (Modular Script)
        * once Execution Starts Remember to close the Performance matrix window to get towards end.


## Steps to Follow (for your own use case) :
    - Label, Training Images using Label-Studio
    - Image File Sample : 
        * image.png
        * masked_images.png
    - Stage1_train/
        - image_folder_name/images/image.png
        - image_folder_name/masks/masks.png
    - Stage1_test/
        - image_folder/images/image.png


## Want to Avoid Checking on Implementation & Directly GOTO Check the results ?
    - After cloning or go through  saved_plots folder to check the result
  

    
