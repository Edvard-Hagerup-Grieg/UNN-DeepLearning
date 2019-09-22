#include <iostream>
#include <fstream> 
#include <iomanip>  
#include "parser.h"
#include "backprop.h"

using namespace std; 


int main()
{
	int L = 100; 

	string path_train_img = "C:\\Users\\Franz\\Desktop\\GitHub\\UNN-DeepLearning\\lab1\\data\\train-images.idx3-ubyte";
	string path_train_lbl = "C:\\Users\\Franz\\Desktop\\GitHub\\UNN-DeepLearning\\lab1\\data\\train-labels.idx1-ubyte";
	double* dataset = read_mnist_images(path_train_img, L);
	int* labels = read_mnist_labels(path_train_lbl, L);

	head(dataset, labels, 10);

	//fit(dataset, labels, 10, 10, 0.1);

	cin.get();
	return 0;
}