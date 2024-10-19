import numpy as np
import re # regular expressions

# this script is intended to adress concerns about the memory layout of input
# matrices when using AMX tiles. This script determines whether the Jx16 matrix
# intended to be stored in the src2 tile register should be laid out in memory 
# as a Jx16 matrix in row-major order, or as the transposition of the Jx16 in
# order to meet the 16x64 tile dimensions

# The result form this test is that the matrix should be laid out within the 16x64 
# tile as a contiguous 64x16 matrix in row-major order.

# When this matrix is loaded into both src1 and src2 registers...
matrix_16_64 = """ 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 
2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 
3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 
4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 
5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 
6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 
7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 
8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 
9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 
10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 
11 11 11 11 11 11 11 11 11 11 11 11 11 11 11 11 11 11 11 11 11 11 11 11 11 11 11 11 11 11 11 11 11 11 11 11 11 11 11 11 11 11 11 11 11 11 11 11 11 11 11 11 11 11 11 11 11 11 11 11 11 11 11 11 
12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 
13 13 13 13 13 13 13 13 13 13 13 13 13 13 13 13 13 13 13 13 13 13 13 13 13 13 13 13 13 13 13 13 13 13 13 13 13 13 13 13 13 13 13 13 13 13 13 13 13 13 13 13 13 13 13 13 13 13 13 13 13 13 13 13 
14 14 14 14 14 14 14 14 14 14 14 14 14 14 14 14 14 14 14 14 14 14 14 14 14 14 14 14 14 14 14 14 14 14 14 14 14 14 14 14 14 14 14 14 14 14 14 14 14 14 14 14 14 14 14 14 14 14 14 14 14 14 14 14 
15 15 15 15 15 15 15 15 15 15 15 15 15 15 15 15 15 15 15 15 15 15 15 15 15 15 15 15 15 15 15 15 15 15 15 15 15 15 15 15 15 15 15 15 15 15 15 15 15 15 15 15 15 15 15 15 15 15 15 15 15 15 15 15 """

# The following output is produces by the AMX tile input when the memory layout in matrix_16_64 
# loaded into both tile registers.
amx_output = """0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
480 480 480 480 480 480 480 480 480 480 480 480 480 480 480 480 
960 960 960 960 960 960 960 960 960 960 960 960 960 960 960 960 
1440 1440 1440 1440 1440 1440 1440 1440 1440 1440 1440 1440 1440 1440 1440 1440 
1920 1920 1920 1920 1920 1920 1920 1920 1920 1920 1920 1920 1920 1920 1920 1920 
2400 2400 2400 2400 2400 2400 2400 2400 2400 2400 2400 2400 2400 2400 2400 2400 
2880 2880 2880 2880 2880 2880 2880 2880 2880 2880 2880 2880 2880 2880 2880 2880 
3360 3360 3360 3360 3360 3360 3360 3360 3360 3360 3360 3360 3360 3360 3360 3360 
3840 3840 3840 3840 3840 3840 3840 3840 3840 3840 3840 3840 3840 3840 3840 3840 
4320 4320 4320 4320 4320 4320 4320 4320 4320 4320 4320 4320 4320 4320 4320 4320 
4800 4800 4800 4800 4800 4800 4800 4800 4800 4800 4800 4800 4800 4800 4800 4800 
5280 5280 5280 5280 5280 5280 5280 5280 5280 5280 5280 5280 5280 5280 5280 5280 
5760 5760 5760 5760 5760 5760 5760 5760 5760 5760 5760 5760 5760 5760 5760 5760 
6240 6240 6240 6240 6240 6240 6240 6240 6240 6240 6240 6240 6240 6240 6240 6240 
6720 6720 6720 6720 6720 6720 6720 6720 6720 6720 6720 6720 6720 6720 6720 6720 
7200 7200 7200 7200 7200 7200 7200 7200 7200 7200 7200 7200 7200 7200 7200 7200 """


def extract_matrix_from_text( input_text: str, num_rows: int, num_columns: int ) -> np.ndarray:
    # extract numbers
    numbers = re.findall(r'-?\d+', input_text)
    # to np array
    return np.array([int(num) for num in numbers]).reshape(num_rows, num_columns)

def transpose_matrix(matrix: np.ndarray) -> np.ndarray:
    return matrix.T

def reshape_matrix(matrix: np.ndarray, num_rows: int, num_columns: int) -> np.ndarray:
    return matrix.reshape(num_rows, num_columns)

def print_matrix(matrix: np.ndarray):
    for row in matrix:
        for element in row:
            print(element, end=" ")
        print()


if __name__ == "__main__":  

    # original matrix
    A = extract_matrix_from_text(matrix_16_64, 16, 64)

    # expected output
    C_expected = extract_matrix_from_text(amx_output, 16, 16)

    # create transposed version
    B_from_transpose_of_A = transpose_matrix(A.copy())

    # create a reshaped version of B
    B_from_reshape_of_A = reshape_matrix(A.copy(), 64, 16)

    # multiply AB
    C_transpose = np.dot(A, B_from_transpose_of_A)
    C_reshape = np.dot(A, B_from_reshape_of_A)


    assert(not np.array_equal(C_expected, C_transpose))

    assert(np.array_equal(C_expected, C_reshape))

    print("Test passed!")