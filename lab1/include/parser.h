#ifndef PARSER_H
#define PARSER_H

#include <fstream> 

using namespace std;


int reverseInt(int i);
double* read_mnist_images(string full_path, int number_of_images_);
int* read_mnist_labels(string full_path, int number_of_images_);
void head(double* dataset, int* labels, int start, int num);


#endif