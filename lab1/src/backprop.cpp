#include <iostream>
#include "parser.h"
#include "neural_network.h"  


double phi_1(double x)
{
	return x;
}

void fit(double* x, int* y, const int L, int ITER, double eta)
{
	const int batch_size = 1;
	const int N = 784;
	const int K = 5;
	const int M = 10;

	double* w_1 = new double[K * N];
	double* w_2 = new double[M * K];

	double* dE_1 = new double[K * N];
	double* dE_2 = new double[M * K];

	double* v = new double[K];
	double* u = new double[M];

	//начальная инициализация весов
	for (int k = 0; k < K; k++)
	{
		//cout << "\nw_1[" << k << "] : ";
		for (int n = 0; n < N; n++)
		{
			w_1[k * N + n] = ((double)rand() / (RAND_MAX)) / 10.0;
			//cout << w_1[k * N + n] << " ";
		}
		//cout << endl;
	}

	for (int m = 0; m < M; m++)
	{
		//cout << "\nw_2[" << m << "] : ";
		for (int k = 0; k < K; k++)
		{
			w_2[m * K + k] = ((double)rand() / (RAND_MAX)) / 10.0;
			//cout << w_2[m * K + k] << " ";
		}
		//cout << endl;
	}


	for (int i = 0; i < ITER; i++)
	{
		for (int l = 0; l < L / batch_size; l++)
		{
			for (int ll = 0; ll < batch_size; ll++)
			{

				if ((y[(l * batch_size + ll) * M + 1] == 1))
					cout << "\nLABEL: 1\n";
				else if ((y[(l * batch_size + ll) * M + 5] == 1))
					cout << "\nLABEL: 5\n";
				else continue;

				double E = 0.0;

				//прямой ход
				cout << "\nv[k] : ";
				for (int k = 0; k < K; k++)
				{
					double SUM_1 = 0.0;
					for (int n = 0; n < N; n++)
						SUM_1 += x[(l * batch_size + ll) * N + n] * w_1[k * N + n];
					v[k] = phi_1(SUM_1);
					cout << v[k] << " ";
				}

				double SUM_EXP = 0.0;
				cout << "\nSUM_2 : ";
				for (int m = 0; m < M; m++)
				{
					double SUM_2 = 0.0;
					for (int k = 0; k < K; k++)
						SUM_2 += v[k] * w_2[m * K + k];
					SUM_EXP += exp(SUM_2);
					u[m] = SUM_2;
					cout << u[m] << " ";
				}
				cout << "\nSUM_EXP = " << SUM_EXP;
				cout << "\nu[m] : ";
				for (int m = 0; m < M; m++)
				{
					E -= y[l * 10 + m] * (u[m] - log(SUM_EXP));
					u[m] = exp(u[m]) / SUM_EXP;
					cout << u[m] << " ";
				}
				cout << "\nE = " << E;

				//обратный ход
				for (int m = 0; m < M; m++)
				{
					//cout << "\nw_2[" << m << "] : ";
					for (int k = 0; k < K; k++)
					{
						dE_2[m * K + k] = -y[(l * batch_size + ll) * M + m] * (1.0 - u[m]) * v[k];
						w_2[m * K + k] -= eta * dE_2[m * K + k];
						//cout << w_2[m * K + k] << " ";
					}
					//cout << endl;
				}

				for (int k = 0; k < K; k++)
				{
					//cout << "\nw_1[" << k << "] : ";
					for (int n = 0; n < N; n++)
					{
						dE_1[k * N + n] = 0.0;
						for (int m = 0; m < M; m++)
							dE_1[k * N + n] -= y[(l * batch_size + ll) * M + m] * (1.0 - u[m]) * w_2[m * K + k] * 1.0 * x[(l * batch_size + ll) * N + n];
						w_1[k * N + n] -= eta * dE_1[k * N + n];
						//cout << w_1[k * N + n] << " ";
					}
					//cout << endl;
				}
			}
			

		}
	}

}