# Arabic-OCR-Project

Dataset.py file to generate a dataset according to detectron2 framework 
where it takes the images inside the dataset folder and generates instances per image 
and then stores the images inside arabic folder splitting them into two sections training and validation.

training section folder contains the notebook needed for the training

testing.py file to test the model after the training where it depends on config.yml file and model_final.pth
that are in detection files.

utils.py file contains the basic functions to use inside the main scripts.
