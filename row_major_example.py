import numpy as np

"""
This script creates a dummy 16x32 and 32x16 matrix, zero pads them to 16x64 and 64x16, computes the 
product (16x16 result) and saves the two as formated C arrays of int8_t type stored in row-major order. 

The purpose of this being to test whether the AMX tiles behave as expected for matrix multiplications 
that do not concern matrices that only contain the same element. 

The original intel example multiplies two 16x64 matrices which contain only twos,resulting in a 16x16 matrix.
However, this introduces some ambiguity as to how exactly this matmul is happening. 

The (little) available documentation online states that matrices are expected to be stored in row-major order 
and that instructions such as _tile_dpbssd are intended to multiply a 16xJ by Jx16 matrix. The documentation
states that both are 16x64, so there is some confusion about how exactly these matrices should be stored in 
memory in order to accomplish the desired matmul. 

The c arrays that are formatted in this script are intended to test whether storing matrix A and B within
the memory layout as row-major order for sizes (16x64) and (64x16) respectively, will result in the correct
16x16 product when performed with the _tile_dpbssd instruction.


NOTE: the matrices are also padded per the algorithm described in amx_fftsum_prototype.py.  
"""



# create a 2D matrix that is 16x32 with each row populated by the row number
def example_matrix():
    return np.array([[j for _ in range(32)] for j in range(16)])    


def format_array_as_c(array: np.ndarray, array_name: str, elements_per_line=16):
    # Ensure array is of integer type
    array = array.astype(int)
    array_str = f"int8_t {array_name}[{array.size}] = {{\n    "
    for i in range(0, array.size, elements_per_line):
        # Convert each element to int, then to string, and join them with comma
        line = ', '.join(str(int(num)) for num in array[i:i+elements_per_line])
        if i > 0:
            array_str += "    "  # Indent new lines
        array_str += line + ",\n"
    array_str = array_str.rstrip(',\n') + "\n};\n"
    return array_str

if __name__ == "__main__" :

    mat_16_32 = example_matrix()
    assert(mat_16_32.shape == (16, 32))
    
    mat_16_32_T = mat_16_32.T
    assert(mat_16_32_T.shape == (32, 16))

    # matmul 
    product = np.matmul(mat_16_32, mat_16_32_T)
    assert(product.shape == (16, 16))

    # create a version of mat_16_32 that has been zero padded to achieve a 16x64 matrix
    mat_16_64 = np.concatenate((mat_16_32, np.zeros((16, 32))), axis=1)

    assert(mat_16_64.shape == (16, 64))

    # next create a version of mat_16_32_T that has been zero padded to achieve a 64x16 matrix
    mat_64_16 = np.concatenate((mat_16_32_T, np.zeros((32, 16))), axis=0)

    # ensure the product of the padded matrices is the same as the product of the original matrices
    product_padded = np.matmul(mat_16_64, mat_64_16)

    print("\n padded 16x64:")
    print(mat_16_64)

    print("\n padded 64x16:")
    print(mat_64_16)

    print("\nProduct of padded 16x64 and padded 64x16 matrices:")
    print(product)

    assert(np.all(product_padded == product))

    # flatten each (storing as row major)
    mat_16_64_flattened = mat_16_64.flatten()
    mat_64_16_flattened = mat_64_16.flatten()
    product_padded_flattened = product_padded.flatten()

    # format as c arrays
    mat_16_64_c_str = format_array_as_c(mat_16_64_flattened, "mat_16_64")
    mat_64_16_c_str = format_array_as_c(mat_64_16_flattened, "mat_64_16")
    product_padded_c_str = format_array_as_c(product_padded_flattened, "product_padded")

    # save to amx_test_inputs dir as c files
    with open("amx_test_inputs_c/row_major_16_64.c", "w") as f:
        f.write(mat_16_64_c_str)
    with open("amx_test_inputs_c/row_major_64_16.c", "w") as f:
        f.write(mat_64_16_c_str)
    with open("amx_test_inputs_c/row_major_product.c", "w") as f:
        f.write(product_padded_c_str)