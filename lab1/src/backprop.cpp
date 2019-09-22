#include <iostream>
#include "parser.h"


double phi_1(double x)
{
	return x;
}


void fit(double* x, int* y, const int L, int ITER, double eta)
{
	const int N = 784;
	const int K = 100;
	const int M = 10;

	double* w_1 = new double[K * N];
	double* w_2 = new double[M * K];

	double* dE_1 = new double[K * N];
	double* dE_2 = new double[M * K];

	double* v = new double[K];
	double* u = new double[M];

	//начальная инициализация весов
	for (int k = 0; k < K; k++)
		for (int n = 0; n < N; n++)
			w_1[k * N + n] = (((double)rand() + 0.01) / (((double)RAND_MAX + 0.01) * 10.0));

	for (int m = 0; m < M; m++)
		for (int k = 0; k < K; k++)
			w_2[m * K + k] = (((double)rand() + 0.01) / (((double)RAND_MAX + 0.01) * 10.0));


	for (int i = 0; i < ITER; i++)
	{
		for (int l = 0; l < L; l++)
		{
			double E = 0.0;

			//прямой ход
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
			for (int m = 0; m < M; m++)
			{
				E -= y[m] * (u[m] - log(SUM_EXP));
				u[m] = exp(u[m]) / SUM_EXP;
			}

			//обратный ход
			for (int m = 0; m < M; m++)
				for (int k = 0; k < K; k++)
				{
					dE_2[m * K + k] = -y[l * M + m] * (1.0 - u[m]) * v[k];
					w_2[m * K + k] += eta * dE_2[m * K + k];
				}

			for (int k = 0; k < K; k++)
				for (int n = 0; n < N; n++)
				{
					dE_1[k * N + n] = 0.0;
					for (int m = 0; m < M; m++)
						dE_1[k * N + n] -= y[l * M + m] * (1.0 - u[m]) * w_2[m * K + k] * 1.0 * x[l * N + n];
					w_1[k * N + n] += eta * dE_1[k * N + n];
				}

		}
	}

}