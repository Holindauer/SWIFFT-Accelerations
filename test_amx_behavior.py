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
# This computation was performed within src/test-amxtile.c
amx_output_1 = """0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
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

# This is what the memory layout for the SWIFFT_PI_key padded partition 1 looks like
# when loaded into the src2 tile register. It is intended to be a 32x16 matrix that 
# has been zero padded up to 64x16 and stored in row-major order. 
src2 = """ -116 78 -118 75 -19 -52 -128 126 22 -12 -60 -88 -115 118 105 78 52 -54 91 96 -60 69 -123 -44 -121 93 3 -8 -116 16 -47 73 -128 -98 105 -74 -115 -128 86 21 -120 -119 -33 -34 -67 -69 -78 -69 4 -64 -46 92 115 49 59 -40 -39 96 61 81 24 -55 -59 89 
0 -56 65 -29 21 -104 124 -27 44 35 44 108 85 -101 -8 -50 46 -87 2 69 -23 36 56 -94 28 -105 104 -19 -95 56 24 58 -106 -15 119 -18 52 112 -88 28 13 37 -97 60 -99 81 -124 60 -25 85 126 -90 -60 100 103 -91 64 -32 125 -52 117 -122 84 -128 
11 -32 87 -38 126 -121 -120 -77 -59 48 68 -54 -18 -5 -63 -22 -79 -71 -128 102 89 -73 -77 41 -108 96 -92 72 -32 -26 -123 -99 -11 51 103 -91 -11 75 9 -57 -96 4 108 35 -128 -89 -49 -113 -36 -25 -88 64 -69 -58 -20 -8 -84 -39 -61 -66 48 -33 5 113 
-75 48 85 -110 -13 -93 -11 68 38 -67 -37 35 -55 91 -100 -106 44 -8 -81 102 -93 125 -110 120 18 -66 -71 125 64 65 -59 -100 -114 -6 -10 -102 -121 -127 88 14 74 121 -7 -124 21 -31 -72 -25 19 47 120 87 -111 70 -25 120 -66 45 33 38 19 -9 110 110 
-89 65 53 -119 -80 56 -38 0 58 121 -109 18 44 100 -42 103 -107 -38 12 81 -60 -107 -126 13 -88 63 -82 -73 48 -22 65 -14 -11 -54 -59 -80 0 28 -63 85 -34 109 -102 -110 -36 60 -124 108 78 -79 46 33 -29 -42 104 31 -50 -71 82 41 42 39 103 119 
69 47 -42 -76 -95 -118 72 -57 45 -94 -98 62 2 -36 124 40 31 -106 27 -117 77 -95 7 27 84 -29 -70 -37 53 126 -95 -15 110 -128 112 47 18 61 -123 78 73 79 -3 -25 125 -77 -52 54 -128 4 15 92 6 116 69 -61 48 -123 84 81 111 56 38 -81 
120 -5 0 -52 28 42 -64 64 39 37 83 -82 5 -39 -42 -83 11 -106 -26 -61 12 122 -95 56 17 114 -66 -50 90 -125 64 -19 -7 5 65 -116 68 116 -26 113 10 -39 99 -52 -56 92 -100 4 -76 -106 0 6 -111 -71 68 55 54 58 -44 -75 60 -26 33 -25 
73 -124 -46 86 120 -116 122 -16 127 56 -127 36 -83 75 83 -11 34 94 -116 -72 18 -23 -100 109 104 64 -7 125 49 -21 86 48 -107 7 127 119 -108 -13 84 -72 -123 34 -128 -64 -21 -23 -125 117 79 31 107 -7 -63 -101 -71 69 -45 66 41 -77 -118 42 -46 -4 
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 """

# This is the output of the using AMX tiles to muliply a 16x64 matrix of ones by the first 64x16
# PI key partition, this is above in the src2 variable.
# This computation was performed within src/test-amxtile-lookup-key.c
amx_output_2 = """
-165 -380 237 74 -363 -326 18 -23 -365 84 -454 -905 365 103 174 656 
-165 -380 237 74 -363 -326 18 -23 -365 84 -454 -905 365 103 174 656 
-165 -380 237 74 -363 -326 18 -23 -365 84 -454 -905 365 103 174 656 
-165 -380 237 74 -363 -326 18 -23 -365 84 -454 -905 365 103 174 656 
-165 -380 237 74 -363 -326 18 -23 -365 84 -454 -905 365 103 174 656 
-165 -380 237 74 -363 -326 18 -23 -365 84 -454 -905 365 103 174 656 
-165 -380 237 74 -363 -326 18 -23 -365 84 -454 -905 365 103 174 656 
-165 -380 237 74 -363 -326 18 -23 -365 84 -454 -905 365 103 174 656 
-165 -380 237 74 -363 -326 18 -23 -365 84 -454 -905 365 103 174 656 
-165 -380 237 74 -363 -326 18 -23 -365 84 -454 -905 365 103 174 656 
-165 -380 237 74 -363 -326 18 -23 -365 84 -454 -905 365 103 174 656 
-165 -380 237 74 -363 -326 18 -23 -365 84 -454 -905 365 103 174 656 
-165 -380 237 74 -363 -326 18 -23 -365 84 -454 -905 365 103 174 656 
-165 -380 237 74 -363 -326 18 -23 -365 84 -454 -905 365 103 174 656 
-165 -380 237 74 -363 -326 18 -23 -365 84 -454 -905 365 103 174 656 
-165 -380 237 74 -363 -326 18 -23 -365 84 -454 -905 365 103 174 656 """

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

def test_row_major_memory_layout():
    """ This test ensures that row-major memory layout is correct for both
    matrices. The C_expected array comes from the output produced by loading
    matrix_16_64 into both src1 and src2 registers."""

    # original matrix
    A = extract_matrix_from_text(matrix_16_64, 16, 64)

    # expected output
    C_expected = extract_matrix_from_text(amx_output_1, 16, 16)

    # create transposed version
    B_from_transpose_of_A = transpose_matrix(A.copy())

    # create a reshaped version of B
    B_from_reshape_of_A = reshape_matrix(A.copy(), 64, 16)

    # multiply AB
    C_transpose = np.dot(A, B_from_transpose_of_A)
    C_reshape = np.dot(A, B_from_reshape_of_A)

    # ensure that the transposed memory layout does not produce the same 
    # result as the row-major layout
    assert(not np.array_equal(C_expected, C_transpose))
    assert(np.array_equal(C_expected, C_reshape))

    print("\nrow major layout is correct for both matrices!\n")


def test_amx_output():
    """ This test ensures that the AMX output is correct for multiplying a 16x64 
     matrix of ones with the generated PI_key lookup. """

    A = np.ones((16, 64), dtype=int)    
    B = extract_matrix_from_text(src2, 64, 16)

    C_expected = extract_matrix_from_text(amx_output_2, 16, 16)

    C = np.matmul(A, B)

    print("Output when manually computed w/ numpy: ")
    print_matrix(C)

    print("\nOutput produced by amx")
    print_matrix(C_expected)

    if not np.array_equal(C, C_expected):
        print("\nAMX tiles not producing correct output...")


if __name__ == "__main__":  

    test_row_major_memory_layout()
    test_amx_output()

    