#include <iostream>
#include "parser.h"
#include "neural_network.h"  


void backprop_step(neural_network *nn, double *x, int* y, double eta, int L)
{

	//head(x, y, 0, 1);

	int N = nn->N;
	int K = nn->K;
	int M = nn->M;

	double err = 0.0, acc = 0.0;

	double* v = new double[L * K];
	double* u = new double[L * M];

	double* dE_1 = new double[K * N];
	double* dE_2 = new double[M * K];

	for (int l = 0; l < L; l++)
	{

		//forward pass
		for (int k = 0; k < K; k++)
		{
			double SUM_1 = 0.0;
			for (int n = 0; n < N; n++)
				SUM_1 += x[l * N + n] * nn->w_1[k * N + n];
			v[l * K + k] = nn->phi_1(SUM_1);
		}

		double max_v = 0.0;
		double sum_exp = 0.0;
		double *SUM_2 = new double[M];
		for (int m = 0; m < M; m++)
		{
			SUM_2[m] = 0.0;
			for (int k = 0; k < K; k++)
				SUM_2[m] += v[l * K + k] * nn->w_2[m * K + k];

			if (SUM_2[m] > max_v)
				max_v = SUM_2[m];
		}

		for (int m = 0; m < M; m++)
		{
			sum_exp += exp(SUM_2[m] - max_v);
			u[l * M + m] = exp(SUM_2[m] - max_v);
		}

		double max_u = 0.0;
		int ans = 0, true_ans = 0;
		for (int m = 0; m < M; m++)
		{
			u[l * M + m] = u[l * M + m] / sum_exp;

			if (u[l * M + m] != 0)
				err -= y[l * M + m] * log(u[l * M + m]);
			if (u[l * M + m] > max_u)
			{
				max_u = u[l * M + m];
				ans = m;
			}
			if (y[l * M + m] == 1)
				true_ans = m;
		}
		if (ans == true_ans)
			acc += 1.0;


		//reverse pass
		for (int k = 0; k < K; k++)
			for (int m = 0; m < M; m++)
			{
				dE_2[m * K + k] = 0.0;
				for (int i = 0; i < M; i++)
					if (i != m)
						dE_2[m * K + k] += y[l * M + i] * u[l * M + i] * v[l * K + k];
				dE_2[m * K + k] *= (1.0 - u[l * M + m]);
			}

		for (int k = 0; k < K; k++)
			for (int n = 0; n < N; n++)
			{
				dE_1[k * N + n] = 0.0;
				for (int m = 0; m < M; m++)
					dE_1[k * N + n] -= y[l * M + m] * (1.0 - u[l * M + m]) * nn->w_2[m * K + k] * 1.0 * x[l * N + n];
			}
	}

	//update the weights
	for (int m = 0; m < M; m++)
		for (int k = 0; k < K; k++)
			nn->w_2[m * K + k] -= eta * dE_2[m * K + k] / (double)L;

	for (int k = 0; k < K; k++)
		for (int n = 0; n < N; n++)
			nn->w_1[k * N + n] -= eta * dE_1[k * N + n] / (double)L;

	cout << "err = " << err / (double)L << "\tacc = " << acc / (double)L << endl;
}