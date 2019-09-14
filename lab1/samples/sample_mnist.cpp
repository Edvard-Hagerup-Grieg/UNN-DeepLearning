#include <iostream>
#include <fstream> 
#include "parser.h"

using namespace std; 


int main()
{
	string path_train_img = "C:\\Users\\Franz\\Desktop\\GitHub\\UNN-DeepLearning\\lab1\\data\\train-images.idx3-ubyte";
	string path_train_lbl = "C:\\Users\\Franz\\Desktop\\GitHub\\UNN-DeepLearning\\lab1\\data\\train-labels.idx1-ubyte";
	double* dataset = read_mnist_images(path_train_img);
	int* labels = read_mnist_labels(path_train_lbl);


	cin.get();
	return 0;
}