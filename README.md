-----Final Project CSI-380 Image Classification Using Rust-----

   -----Griffin Smith & Glen Kelley-----
          

This project uses template matching to classify images in rust. 
The program creates templates of each class and compares and classifies images based off the templates
The program offers the ability to use a sequential implementation or a parallel implementation with 10 templates per class or 1 template per class
We measured the accuracy, throughtput, and total time that it takes each implementation to run
This project uses the MNIST handwritten digits dataset which can be downloaded using the link below 

--------------------------------------------------------------------------------------------------------

Go to https://www.kaggle.com/datasets/hojjatk/mnist-dataset and download the handwritten digits dataset as a zip folder

Take these four files (the files not the folders):

t10k-images.idx3-ubyte

t10k-labels.idx1-ubyte

train-images.idx3-ubyte

train-labels.idx1-ubyte

and drag them into the data folder included in the program

after that the dataset should be all set and ready for you to use

--------------------------------------------------------------------------------------------------------

Any version of Rust over 1.60 should run the program
to run use cargo --run

To run you should at least have 4gb of ram and a cpu that has at least two cores

