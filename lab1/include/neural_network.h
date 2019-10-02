#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H


class neural_network
{
public:
	int N;
	int K;
	int M;

	double* w_1;
	double* w_2;

	neural_network(int N, int K, int M);

	double phi_1(double x);
	void phi_2(double* x, int n);
	
	void fit(double* x, int* y, const int L, int batch_size, int iter, double eta);
	void predict(double* x, int* y, int L);
};


#endif