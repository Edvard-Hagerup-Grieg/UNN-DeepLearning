#include <iostream>
#include <fstream> 
#include <algorithm>

using namespace std;


int reverseInt(int i)
{
	unsigned char c1, c2, c3, c4;

	c1 = i & 255;
	c2 = (i >> 8) & 255;
	c3 = (i >> 16) & 255;
	c4 = (i >> 24) & 255;

	return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + (int)c4;
}

double* read_mnist_images(string path, int number_of_images_)
{
	ifstream file(path, ios::binary);
	if (file.is_open())
	{
		int magic_number = 0;
		int number_of_images = 0;
		int n_rows = 0;
		int n_cols = 0;

		file.read((char*)& magic_number, sizeof(magic_number));
		magic_number = reverseInt(magic_number);

		file.read((char*)& number_of_images, sizeof(number_of_images));
		number_of_images = reverseInt(number_of_images);
		number_of_images = min(number_of_images, number_of_images_);

		file.read((char*)& n_rows, sizeof(n_rows));
		n_rows = reverseInt(n_rows);

		file.read((char*)& n_cols, sizeof(n_cols));
		n_cols = reverseInt(n_cols);
	

		int image_size = n_rows * n_cols;
		double* arr = new double[image_size*number_of_images];
		for (int i = 0; i < number_of_images; ++i)
			for (int r = 0; r < n_rows; ++r)
				for (int c = 0; c < n_cols; ++c)
				{
					unsigned char temp = 0;
					file.read((char*)& temp, sizeof(temp));
					arr[i*n_cols*n_rows + r * n_cols + c] = (double)temp / 255;
				}
		return arr;
	}
	return NULL;
}

int* read_mnist_labels(string path, int number_of_labels_)
{
	ifstream file(path);
	if (file.is_open())
	{
		int magic_number = 0;
		int number_of_labels = 0;

		file.read((char *)&magic_number, sizeof(magic_number));
		magic_number = reverseInt(magic_number);

		file.read((char *)&number_of_labels, sizeof(number_of_labels));
		number_of_labels = reverseInt(number_of_labels);
		number_of_labels = min(number_of_labels, number_of_labels_);

		int* labels = new int[number_of_labels * 10];
		for (int i = 0; i < number_of_labels; i++) 
		{
			for (int j = 0; j < 10; j++)
				labels[i * 10 + j] = 0;

			unsigned char tmp = 0;
			file.read((char*)&tmp, 1);
			labels[i*10 + int(tmp)] = 1;
		}
		return labels;
	}
	return NULL;
}

void head(double* dataset, int* labels, int start, int num)
{
	for (int k = start; k < start + num; k++)
	{
		int l = 0;
		for (int n = 0; n < 10; n++)
			if (labels[k * 10 + n] == 1)
				l = n;
		cout << "\nLABEL: " << l << endl;

		for (int i = 0; i < 28; i++)
		{
			for (int j = 0; j < 28; j++)
				if (dataset[k * 784 + i * 28 + j] > 0)
					cout << 1 << " ";
				else
					cout << 0 << " ";
			cout << endl;
		}
	}
}