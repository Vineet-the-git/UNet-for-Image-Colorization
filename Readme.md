# Colourization of Grayscale images using UNet

## How to setup the project?

Project Organization
------------
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   └── raw            <- Folder containg the original coloured images.
    │
    ├── experiments        <- Trained and serialized models saved during the training after 5 epochs.
    │
    ├── predictions        <- A video showing improvements in colorization over 30 epochs. 
    |
    ├── references         <- Contains a readme file containg all the refrences.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Figures and Plots.
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment.
    │
    ├── unet               <- Architecture of UNET.
    |   ├── unet_model.py  <- UNET model
    |   └── unet_parts.py  <- Parts of UNET
    |
    ├── dataloader.py      <- Dataloader for loading train and validation data.
    |
    ├── inference.py       <- Takes path of model and grayscale image and generates a colour image from it.
    |
    ├── train.py           <- Train, validate and save the model.
    |
    └── saving.py          <- Saving the fig, model checkpoints. 
--------

1. Install the requirements  using `pip install -r requirement.txt`.        
2. To train the model and validate the model:
    -   Run: `python train.py`
3. To test the model on particular grayscale image on a particular model checkpoint:
    -   Run:  `python inference.py --p_img "path_to_image" --p_model "path_to_model"`



