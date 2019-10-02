#include "neural_network.h"
#include "backprop.h"
#include <cmath>
#include <iostream>


using namespace std;

neural_network::neural_network(int n, int k, int m)
{
	N = n;
	K = k;
	M = m;

	w_1 = new double[K * N];
	w_2 = new double[M * K];
}

double neural_network::phi_1(double x)
{
	return x;
}

void neural_network::phi_2(double* x, int n)
{
	double EXP_SUM = 0.0;
	for (int i = 0; i < n; i++)
		EXP_SUM += exp(x[i]);
	for (int i = 0; i < n; i++)
		x[i] = exp(x[i]) / EXP_SUM;
}

void neural_network::fit(double* x, int* y, const int L, int batch_size, int iter, double eta)
{
	double* dE_1 = new double[K * N];
	double* dE_2 = new double[M * K];

	double* v = new double[K];
	double* u = new double[M];

	//начальная инициализация весов
	for (int k = 0; k < K; k++)
		for (int n = 0; n < N; n++)
			w_1[k * N + n] = ((double)rand() / (RAND_MAX)) / 2.0;

	for (int m = 0; m < M; m++)
		for (int k = 0; k < K; k++)
			w_2[m * K + k] = ((double)rand() / (RAND_MAX)) / 2.0;


	for (int i = 0; i < iter; i++)
	{
		cout << "\nEPOCH : " << i << "/" << iter << endl;
		for (int l = 0; l < L / batch_size; l++)
		{
			cout << "items : " << (l + 1) * batch_size << "/" << L << " :\t";
			backprop_step(this, x + l * N, y + l * M, eta, batch_size);
		}
	}
}

void neural_network::predict(double* x, int* y, int L)
{
	double err = 0.0;
	double acc = 0.0;
	double* v = new double[K];
	double* u = new double[M];

	for (int l = 0; l < L; l++)
	{
		for (int k = 0; k < K; k++)
		{
			double SUM_1 = 0.0;
			for (int n = 0; n < N; n++)
				SUM_1 += x[l * N + n] * w_1[k * N + n];
			v[k] = phi_1(SUM_1);
		}

		double SUM_EXP = 0.0;
		for (int m = 0; m < M; m++)
		{
			double SUM_2 = 0.0;
			for (int k = 0; k < K; k++)
				SUM_2 += v[k] * w_2[m * K + k];
			SUM_EXP += exp(SUM_2);
			u[m] = SUM_2;
		}

		double MAX = 0.0;
		int ANS = 0;
		int LBL = 0;
		for (int m = 0; m < M; m++)
		{
			err -= y[l * 10 + m] * (u[m] - log(SUM_EXP));
			u[m] = exp(u[m]) / SUM_EXP;

			if (u[m] > MAX)
			{
				MAX = u[m];
				ANS = m;
			}
			if (y[l * 10 + m] == 1)
				LBL = m;
		}
		cout << "\nLABEL: " << LBL << "\tANSWER: " << ANS;
		if (ANS == LBL)
			acc += 1.0;
	}
	cout << "\nERR = " << err / L << "\tACC = " << acc / L << endl;
}