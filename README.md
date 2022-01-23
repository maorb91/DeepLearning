Run the VAE_Anomaly_Detection.ipynb notebook in order to train simple VAE,load pretrained VAEs in and see the results of each one of the VAEs

to tarin VAE run train.py this will save the model to the project folder

if you want to change the vae parameters(number of layers, latent space dimension, beta, and alpha) edit the tarin.py main function 

if you want to change the vae architecture(the type of layers and etc) edit the vae.py

run utils.py - saves the result of each one of the models in the "model" folder and save the result to 3 different CSV file(one for each type of VAE)

measurements.py - contains many measurements function used by the jupyter notebook in this git and by utils.py


minimum requirements :  
python == 3.6  
scikit-learn ==  1.0.1  
tensorflow == 2.3.0  
tensorboard == 2.6.0  
seaborn ==  0.11.2  
numpy ==  1.19.2  
pandas ==  1.3.4  
matplotlib ==  3.5.0  
keras ==  2.4.3
