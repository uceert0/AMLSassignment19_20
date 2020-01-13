https://drive.google.com/drive/folders/13cCeHVNvrxEKiC9Wfi7kF0HjO_Wf3c2a


This project contains the codebase that was implemented to solve the problems of a) gender, b) emotion (smile), c) face shape, and d) eyes’ colour classification.

Development was performed by using the conda as the python distribution under Windows. In order to be able execute the codebase please follow the following steps.

1. Download and install Conda (https://docs.conda.io/en/latest/miniconda.html)

2. Create a new environment by using the provided environment.yml file (conda env create -f environment.yml)

3. Activate the created enviroment

4. Move the provided folders celeba (with img subfolder and label.csv) and catoon_set (with img subfolder and label.csv) under the folder Datasets/initial/. Please do not modify the folder names

5. Move the provided folders celeba_test(with img subfolder and label.csv) and catoon_set_test (with img subfolder and label.csv) under the folder TestSets/initial/ Please do not modify the folder names
   After moving the folders to the correct dir, the folder structure should be the following:
	- Datasets/initial/celeba/img
	- Datasets/initial/celeba/labels.csv
	- Datasets/initial/cartoon_set/img
	- Datasets/initial/cartoon_set/labels.csv
	- TestSets/initial/celeba_test/img
	- TestSets/initial/celeba_test/labels.csv
	- TestSets/initial/cartoon_set_test/img
	- TestSets/initial/cartoon_set_test/labels.csv

6. Execute the main.py by using python main.py. The function for each task will generate and save the preprocessed datasets (under Datasets/generated/<task>/) and will extract and save the features
   (under Datasets/features/<task>/). Then will start training the model and subsquently will peform the inference. After training and testing each model, the preprocessing, 
   feature extraction and inference will be executed for the Secret Test Sets. Finally, the accuracies across the different tasks and sets will be displayed.

