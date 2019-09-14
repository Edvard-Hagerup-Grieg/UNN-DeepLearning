#ifndef PARSER_H
#define PARSER_H

#include <fstream> 

using namespace std;


int reverseInt(int i);
double* read_mnist_images(string full_path);
int* read_mnist_labels(string full_path);


#endif