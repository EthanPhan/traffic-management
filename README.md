# traffic-management
Code for the traffic management challenge from Grab in 2019.

This repo contains only the code for for prediction. Training script will be added in the future

## Model descriptions
The model used for this project is a variant wavenet.

Number of timestep used for prediction: 14 (days) * 96 (timestep per day) = 1344.

Number of prediction output: 5 future timestep

Features: demand and normalized time of day

## Install
Install all the packages needed:

`pip install -r requirements.txt`

## How to run
1. From the root directore, create `data/predict/` directory and put the csv file we want to make prediction on in there
2.  From the root, execute this: 

`python predict.py --use_prophet --num_thread 4 --batch_size 64`

#### arguments:
  `--use_prophet` use facebook's prophet model instead of 
  
  `--num_thread ` number of thread you want to use to speed up prediction
  
  `--batch_size` batch size. Only applicable when predict using keras model
  
 
***Note:*** Currently , prophet yeild better result compared to the wavenet model but it's significantly slower because we have to retrain the model for every location. Which model to use is up to you


The output of the prediction is in `output/` directory
