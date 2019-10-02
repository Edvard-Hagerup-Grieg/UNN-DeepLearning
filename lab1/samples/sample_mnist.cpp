#include <iostream>
#include <fstream> 
#include <iomanip>
#include "omp.h"
#include "parser.h"
#include "neural_network.h"

using namespace std; 


int main()
{
	double start = 0, finish = 0;

	int batch_size = 50;
	int train = batch_size * 1;
	int test = 10;
	int L = train + test; 

	string path_train_img = "C:\\Users\\Franz\\Desktop\\GitHub\\UNN-DeepLearning\\lab1\\data\\train-images.idx3-ubyte";
	string path_train_lbl = "C:\\Users\\Franz\\Desktop\\GitHub\\UNN-DeepLearning\\lab1\\data\\train-labels.idx1-ubyte";
	double* dataset = read_mnist_images(path_train_img, L);
	int* labels = read_mnist_labels(path_train_lbl, L);

	//head(dataset, labels, 0, 10);

	neural_network model(784, 50, 10);
	start = omp_get_wtime();
	model.fit(dataset, labels, train, batch_size, 500, 1E-4);
	finish = omp_get_wtime();
	model.predict(dataset + train, labels + train, test);

	cout << "\n\nTime : " << (finish - start);
	cin.get();
	return 0;
}