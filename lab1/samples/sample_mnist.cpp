#include <iostream>
#include <fstream> 
#include <iomanip>  
#include "parser.h"
#include "neural_network.h"

using namespace std; 


int main()
{	
	int train = 1;
	int test = 30;
	int L = train + test; 

	string path_train_img = "C:\\Users\\Franz\\Desktop\\GitHub\\UNN-DeepLearning\\lab1\\data\\train-images.idx3-ubyte";
	string path_train_lbl = "C:\\Users\\Franz\\Desktop\\GitHub\\UNN-DeepLearning\\lab1\\data\\train-labels.idx1-ubyte";
	double* dataset = read_mnist_images(path_train_img, L);
	int* labels = read_mnist_labels(path_train_lbl, L);

	//head(dataset, labels, 10);

	neural_network model(784, 10, 10);
	model.fit(dataset, labels, train, 1, 0.0001);
	model.predict(dataset + train, labels + train, test);

	cin.get();
	return 0;
}