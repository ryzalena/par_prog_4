import numpy as np
import sys


def read_matrix(filename):
    with open(filename, 'r') as f:
        shape = f.readline().strip().split()
        rows, cols = int(shape[0]), int(shape[1])
        matrix = np.array([list(map(int, f.readline().strip().split())) for _ in range(rows)])
    return matrix


def verify_multiplication(result_file, matrixA_file, matrixB_file):
    result_matrix = read_matrix(result_file)
    A = read_matrix(matrixA_file)
    B = read_matrix(matrixB_file)

    expected_result = np.dot(A, B)

    if np.array_equal(result_matrix, expected_result):
        print("Верификация успешна: результат верен.")
    else:
        print("Верификация не удалась: результат неверен.")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Использование: python verify_results.py <result_file> <matrixA_file> <matrixB_file>")
        sys.exit(1)

    result_file = sys.argv[1]
    matrixA_file = sys.argv[2]
    matrixB_file = sys.argv[3]

    verify_multiplication(result_file, matrixA_file, matrixB_file)
