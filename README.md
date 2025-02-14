# satelliteCNN

This is the reference implementation of the following paper:

[Sarmadi, Hamid, Ibrahim Wahab, Ola Hall, Thorsteinn Rögnvaldsson, and Mattias Ohlsson. “Human Bias and CNNs’ Superior Insights in Satellite Based Poverty Mapping.” Scientific Reports 14, no. 1 (October 2, 2024): 22878. https://doi.org/10.1038/s41598-024-74150-9.](https://www.nature.com/articles/s41598-024-74150-9)

## Python Package Requirements

- Tensorflow 2.10
- pandas
- pillow
- scikit-learn
- scikit-image
- matplotlib
- seaborn
- fastparquet
- statsmodels

# Quick start

To calculate the results given the test images provided with the code you just need to run:

``source evaluation_classification.sh``

After running the code some of the results will be shown in the terminal. The code also produces some plots that are save in the ``plots`` folder.

# Backbone Training

You do not need to train the backbone, you can use the model already provided with this code in the ``models`` folder.
However, in case you want to do the training yourself, you have to download the training data.
When you train a new model it will be in the models folder and you will need to change the model path in the evaluate_classification.sh file.

## Downloading the training data

You can download the training data from [https://doi.org/10.7910/DVN/PPWAFG](https://doi.org/10.7910/DVN/PPWAFG).
After downloding all the files extract them to a folder named "image_data" in the root directory.

## Running the training code

To run the training code you just need use the following command:

``python backbone_training/train_backbone.py``


## Please cite our paper if you are using our code
```
@article{sarmadi_human_2024,
	title = {Human bias and {CNNs}’ superior insights in satellite based poverty mapping},
	volume = {14},
	copyright = {2024 The Author(s)},
	issn = {2045-2322},
	url = {https://www.nature.com/articles/s41598-024-74150-9},
	doi = {10.1038/s41598-024-74150-9},
	language = {en},
	number = {1},
	urldate = {2024-12-09},
	journal = {Scientific Reports},
	author = {Sarmadi, Hamid and Wahab, Ibrahim and Hall, Ola and Rögnvaldsson, Thorsteinn and Ohlsson, Mattias},
	month = oct,
	year = {2024},
	note = {Publisher: Nature Publishing Group},
	keywords = {Computer science, Socioeconomic scenarios, Environmental economics},
	pages = {22878}
}
```
