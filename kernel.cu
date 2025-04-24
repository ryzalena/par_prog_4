#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>
#include <iostream>

__global__ void multiplyMatricesKernel(int* matrix1, int* matrix2, int* result_matrix, int size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < size && col < size) {
        int value = 0;
        for (int k = 0; k < size; ++k) {
            value += matrix1[row * size + k] * matrix2[k * size + col];
        }
        result_matrix[row * size + col] = value;
    }
}

void generateRandomMatrix(int* matrix, int size) {
    for (int i = 0; i < size * size; ++i) {
        matrix[i] = rand() % 100;
    }
}

void multiplyMatrices(int* matrix1, int* matrix2, int* result_matrix, int size) {
    int* d_matrix1;
    int* d_matrix2;
    int* d_result_matrix;
    int rank = 128;

    cudaMalloc(&d_matrix1, size * size * sizeof(int));
    cudaMalloc(&d_matrix2, size * size * sizeof(int));
    cudaMalloc(&d_result_matrix, size * size * sizeof(int));

    cudaMemcpy(d_matrix1, matrix1, size * size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_matrix2, matrix2, size * size * sizeof(int), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(rank, rank);
    dim3 numBlocks((size + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (size + threadsPerBlock.y - 1) / threadsPerBlock.y);

    multiplyMatricesKernel << <numBlocks, threadsPerBlock >> > (d_matrix1, d_matrix2, d_result_matrix, size);

    cudaMemcpy(result_matrix, d_result_matrix, size * size * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_matrix1);
    cudaFree(d_matrix2);
    cudaFree(d_result_matrix);
}

void writeMatrixToFile(const char* filename, int* matrix, int size) {
    FILE* file = fopen(filename, "w");
    if (file == NULL) {
        printf("Не удалось открыть файл: %s\n", filename);
        exit(1);
    }

    fprintf(file, "%d\n", size);
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            fprintf(file, "%d ", matrix[i * size + j]);
        }
        fprintf(file, "\n");
    }

    fclose(file);
}

void writeTimeToFile(double computation_time, const char* filename) {
    FILE* file = fopen(filename, "a");
    if (file == NULL) {
        printf("Не удалось открыть файл: %s\n", filename);
        exit(1);
    }

    fprintf(file, "%.5f\n", computation_time);
    fclose(file);
}

void writeTaskSizeToFile(int size, long long task_size, const char* filename) {
    FILE* file = fopen(filename, "a");
    if (file == NULL) {
        printf("Не удалось открыть файл: %s\n", filename);
        exit(1);
    }

    fprintf(file, "%d\n%lld\n", size, task_size);
    fclose(file);
}

int main() {
    setlocale(LC_ALL, "ru_RU.UTF-8");

    int N = 500;
    const char* file_stat = "result.txt";

    srand((unsigned int)time(NULL));

    while (N <= 3000) {
        long long task_size = (long long)N * N * N;
        writeTaskSizeToFile(N, task_size, file_stat);

        printf("Размер матриц %dx%d\n", N, N);

        for (int i = 0; i < 10; ++i) {
            int* matrix1 = (int*)malloc(N * N * sizeof(int));
            int* matrix2 = (int*)malloc(N * N * sizeof(int));
            int* result_matrix = (int*)malloc(N * N * sizeof(int));

            if (!matrix1 || !matrix2 || !result_matrix) {
                printf("Ошибка выделения памяти\n");
                exit(1);
            }

            generateRandomMatrix(matrix1, N);
            generateRandomMatrix(matrix2, N);

            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);

            cudaEventRecord(start);
            multiplyMatrices(matrix1, matrix2, result_matrix, N);
            cudaEventRecord(stop);

            cudaEventSynchronize(stop);

            float milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, start, stop);

            printf("Матрицы перемножены.\n");
            printf("Время умножения матриц: %.5f мс\n", milliseconds);

            // записываем время в файл с точностью до 5 знаков после запятой
            writeTimeToFile((double)milliseconds, file_stat);

            free(matrix1);
            free(matrix2);
            free(result_matrix);

            cudaEventDestroy(start);
            cudaEventDestroy(stop);
        }

        N += 500;
    }

    return 0;
}