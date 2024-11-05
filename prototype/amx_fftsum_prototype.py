import numpy as np
import re # regular expression

"""
This file contains a protype of a proposed algorithm for using AMX tiles to speed up the fftsum
component of the LibSWIFFT library. The fftsum component (from a perspective slightly abstracted 
from the order it performs this) computes the hadamard product of two 32x64 matrices (PI_key and 
fftout in the context of LibSWIFFT) followed by a sum along the column dimmensions, resulting in 
a 1x64 vector output.

The algorithm preseted below is intended to utilize intel AMX tile instructions to speed up the 
process of the above described computation. It does this with the following steps:

Given input fftout and PI_key (both 32x64):

1.) Transpose fftout resulting in a (64x32)

2.) Evenly partition fftout_transpose into 4 16x32 matrices

3.) Evenly partition PI_key into 4 32x16 matrices

4.) Zero pad the partitions from 16x32 to 16x64 and 32x16 to 64x16 for the
    fftout and PI_key respectively. The AMX tiles work on tiles that are 
    statically size at 16x64, however, this is different than the matrix 
    stored within them.

5.) Compute the matmul of each likewise partitions (F.T_1 * P_1, F.T_2 * P_2, ... )

6.) concatenate the diagonals of the above products to produce the final output

"""

ORIGINAL_C_PI_KEY = "const SWIFFT_ALIGN int16_t SWIFFT_PI_key_copy[SWIFFT_M*SWIFFT_N] = {	-116,  78,-118,  75, -19, -52,-128, 126,	  22, -12, -60, -88,-115, 118, 105,  78,	  50,-108,  29, -49, 114,  34,  85, 117,	  67,-109,  86,  -1,  25,  49,-124,  93,	  95,  36,  68, -26, -46, 102,-106, 128,	 -33, 117, -64,  27, 102, -70,   7, 105,	  45,-127, 108, 124, -86,-106, -68, 128,	 -39,-123, -24, -92,  14, -56,-112,-123,	  52, -54,  91,  96, -60,  69,-123, -44,	-121,  93,   3,  -8,-116,  16, -47,  73,	   6,  92,  58,  74, -83,   6,  -3,  91,	 -56, 107, 110,  76, 103,  11,  73,  16,	  34, -48,   7, 127,-111,  -3,  95, -81,	  57,  13, 108, -12,  77,  92, -71, 117,	 124,  97, 105, 118,  34,  74, -52, 122,	 -22,  53,  94, -19, -47, -30, -74,  11,	-128, -98, 105, -74,-115,-128,  86,  21,	-120,-119, -33, -34, -67, -69, -78, -69,	  -1,  25, -40, -81,  36, -81, -19, 127,	 -97, -47,-102,-109,-125,   0,  54, 127,	-112,   6,  46,  85, -14,  95, -84, 123,	 -79, -50, -46, -74, -33, -84,-111,  35,	  71, 114,  50,  22, -82,   1,  28,  19,	 112,-128,  21,  34, -96, -98, 115,  52,	   4, -64, -46,  92, 115,  49,  59, -40,	 -39,  96,  61,  81,  24, -55, -59,  89,	  45, 128,   8,  51,  -4,  87, -86,  35,	   4, -69, -86,  10,   3,-120, -19,  73,	  19, -49, 124, -94, 103, -80,-102,-110,	  46,  84,  -4, -24, -86, -16, -46, -40,	 -98,  48,  96,  79, -20,  18, -86, -31,	  99,   1,  97, -62, -41, -94, -59,  95,	   0, -56,  65, -29,  21,-104, 124, -27,	  44,  35,  44, 108,  85,-101,  -8, -50,	  26, -35,-126,   1,  60, -15, -60,-107,	 -76,  19, 116, -44,  75,  98, 124, -17,	 123, -50,  62,  -2,  60,-114, -70,-100,	-118,   9,  12, 104,  89,  49, -64,-111,	 104, -61, -76,  82, -59,  -4, -65, -66,	  -2, 122, -45, 104,  47,  20,-125, -49,	  46, -87,   2,  69, -23,  36,  56, -94,	  28,-105, 104, -19, -95,  56,  24,  58,	  38,-107, -64,  -3,  -4, 125, -84,  35,	  73, 126, -10, -18, -41,   6, -58,  15,	  90,  12,  97, 122,   9,  84, -50, 127,	 -38,  72,  58,  30,  29, -75,  41, -65,	 -22,  -9, -20,  74,  72, -81, -47,  -5,	  45,  64, -92,  87, -55, -16, -21, -34,	-106, -15, 119, -18,  52, 112, -88,  28,	  13,  37, -97,  60, -99,  81,-124,  60,	  16,-112,  -8, -65, -84, -40, -43,  93,	-116, -73,  54,  34, -96, 104,-100,  95,	  38,-124, -39, -30, -46, -76,   9,  66,	-120,-114,  77,  33,  -9, -98,   4,  55,	 -29,  48,  99, -38, -35, -73,  15,  36,	  -3,  -1,-100, -20,  87,-118, -48, 113,	 -25,  85, 126, -90, -60, 100, 103, -91,	  64, -32, 125, -52, 117,-122,  84, 128,	 -26, 112,  90, -16,  28,  22, -47,-110,	 -71,  49, -27,  21, 108,  39, -63,  47,	 123, -58, 107, 114,  30, -47,  -7,-114,	  59,-101,-126,-124, -36,  27,  76,  99,	 -49,  -7,  78,  12, -46,-116,  95,  81,	 -62, 106,   8, -25,-107, -45, -52, -36,	  11, -32,  87, -38, 126,-121,-120, -77,	 -59,  48,  68, -54, -18,  -5, -63, -22,	-115,-120, -83, -85, -67,-112,  -7, -36,	 -75, -53,   1, -62,-127,-104,  83, -16,	 -96, -18, -46,-119,  11, -88,-102, -12,	 -83,  49,  10, -91,  16,-127, -76,-118,	 -35, -35, 112,  99, 124,  94,  51, -14,	-124, -63, -13,-121,  35,  -9, -56, -80,	 -79, -71,-128, 102,  89, -73, -77,  41,	-108,  96, -92,  72, -32, -26,-123, -99,	 -58,  28,  -8,  16, -32, -62,  10, -47,	 -93,  -5,-119,   8,  35,-105, -44, -58,	  82, 116,  97, -27,  63, -58, -16,  35,	  79, 120,  54, -83,  67, 112,   1,  76,	  69, -35, -63,  96,  82,  94,  25, -29,	 -61,-112,-102,-121, -29, -23,  46, 101,	 -11,  51, 103, -91, -11,  75,   9, -57,	 -96,   4, 108,  35,-128, -89, -49,-113,	  50,  14,  13, -37,  41,-125, 122, 127,	 -63,   9, -25, -23, 107,  28, -70,   8,	  51,-116,  97, -36, -32,   9, 113, -87,	 -91, 102,-122,  22, -26, -72, -30, -70,	 110,-112,  -6,-111,  76,  22,-111, -29,	   7,  53,  64,  25,  62, -59,-127, -67,	 -36, -25, -88,  64, -69, -58, -20,  -8,	 -84, -39, -61, -66,  48, -33,   5, 113,	 100, -91, -97,  21, -66, -60,  61, -95,	-108, -86, -17, -74,-128, -26, 123, -53,	 -65, -78,-123,  15,  47, -96,-115, -80,	 -18, -23, -71, -20, -26,  53, -49,  95,	-111,  36, -32, -26,  89,-115,  93,  -9,	-120, 124,  83,  39,  69,  77,  89, -49,	 -75,  48,  85,-110, -13, -93, -11,  68,	  38, -67, -37,  35, -55,  91,-100,-106,	 -56, -17, -72, -39,   4,-105,   2,-125,	 -80,  88, -67, -61, -28,  74, -37,-122,	-120, -61,  11,  47,   5,  -6, 106,-113,	 -94,  60, -35, 127,  52,  57, -55, 102,	  64,-117, 110, -51,  23, -75,  39, -12,	   1, -94,-100, -71, -94,  80,   7, -27,	  44,  -8, -81, 102, -93, 125,-110, 120,	  18, -66, -71, 125,  64,  65, -59,-100,	 -93, -44,  95,  61,  13, -76, -49,  91,	 -15, -60, -99,  34,  98, -88,  91,  14,	  17,  93,-100,  17,  65,  30, -74,   6,	-118,  58,  -2, 108, 100,-121, -48,-113,	 -93,   6, -20,  33, -47, 110,  57, 126,	 -60,-121, 125, -13, -92,-106, -89,   3,	-114,  -6, -10,-102,-121,-127,  88,  14,	  74, 121,  -7,-124,  21, -31, -72, -25,	 118,-125,  89,  64, -53, -96,   2,  70,	 -33, -98,  35, -53, 123, -77,  13,  52,	 -26,  57,  25,  78,  66,  69,  97,  42,	 -59,  84, -81,  59,   8, -25, 125,-123,	 -64,   2, -25, 109, -41,  69,  90,-115,	  32,  38,  -8,  37,  75, -77, -73, -69,	  19,  47, 120,  87,-111,  70, -25, 120,	 -66,  45,  33,  38,  19,  -9, 110, 110,	  44,  64,   2,  84, -13, -29,  -5, -29,	 -87, 123,  38,-113, -44,-113, -86, -45,	 -14,  87, -68,  46, 128, 110,  84,  77,	  65, -74,  61, -73, 101,  44, -89,  68,	  14, 106, 105,   8, -30, -46, -91,  39,	-105,  43,  52,  -3, -60,  55, 119,  89,	 -89,  65,  53,-119, -80,  56, -38,   0,	  58, 121,-109,  18,  44, 100, -42, 103,	-112, -28, 117, -61,  91,  89, 113,-114,	 -85, -18,  -8, -73,-103,  39, 112,  65,	 -53,  42,  84,  38,-102,-106,-106,  16,	 100,  87, -83, -95,-112,-110,-108, -71,	 -20,-112,-123,-113, -59, -22, -44, -94,	  48, -27,  24,  47,  57,  71, 127,   0,	-107, -38,  12,  81, -60,-107,-126,  13,	 -88,  63, -82, -73,  48, -22,  65, -14,	-108, -57, -94,  -3, -55, 114, -10,  67,	-114,  -7, 126, -29,  80,-127, -41, -43,	  36,   2, -27,  33, 119, 125,   3,-115,	 -20, 100,   3,-105, -60, -83, -13,-128,	 -25,  30, -51, -58,  39, -47, -37,  43,	 -20, -36, -56,  54, -78,  42,  28,-124,	 -11, -54, -59, -80,   0,  28, -63,  85,	 -34, 109,-102,-110, -36,  60,-124, 108,	-100,  -3,  26,  75,-100, -72,  49,-115,	  31,-120,  71,  43,  63,  64, -20,-109,	 -20, -85, -98, -97,-102,  -3, -23, -33,	-117, -64, 114,-117,  62, 109,-121,  39,	  -2,   8, -99,-111, 128,  49, -35,  96,	  57, -48, -77,  -8, -55, 127, 113, -26,	  78, -79,  46,  33, -29, -42, 104,  31,	 -50, -71,  82,  41,  42,  39, 103, 119,	 123,-124, -14,  -3, -19,-101,  90, -71,	  37, -45,  33, 107,  -5,  51, -80,  36,	 -20,  76, -98, -12,  93, -43,  97,  56,	 -67,  38, -97,  94, 105, -35, -37, -99,	  49,  16, -66,  52, 120,  87, -78,   2,	  27,-113, -34, -27, -73,   6,-128, -30,	  69,  47, -42, -76, -95,-118,  72, -57,	  45, -94, -98,  62,   2, -36, 124,  40,	 -98, -15,  35, -49, -78, -91,  98,  67,	 -79,  68,-114, -32, -79,-111, -70, -98,	  57,  66, -81, -65, -21,  -7, -89, -33,	 122,  43, -98, 120,-124, -92, 122,  64,	  87,  74, -96, -16,   9,  87,  90,  24,	  -2, 113, -54, -37,  57,-118, -60, -98,	  31,-106,  27,-117,  77, -95,   7,  27,	  84, -29, -70, -37,  53, 126, -95, -15,	  84, -76, -34, 103,  86, -80, -50,  31,	-117,  18, -50,  -1, -56, -91,  96,  23,	 -24, 103, -60,  84, -96,  75,  59,-108,	-119,-103, 119,  92,  16,  53, 116,  97,	 -37, 114,  35,  45,  77, -48,  40, -61,	  71,  22,  81, -79, 110,  14,   3, -77,	 110,-128, 112,  47,  18,  61,-123,  78,	  73,  79,  -3, -25, 125, -77, -52,  54,	 -37, 119,  63,  89, -76,  52,  77, 109,	-106,  77,  80, -50,-113,  25,  20,   6,	 -49,  47, -56, -51, -65,  14,  73, -81,	  -1, -56, -50,  87, -41,  60,  56,  73,	  92, -14, -78, 113,  49,  59,  55, -89,	 121,-120,  69,-103,  95,  57, -70,  47,	-128,   4,  15,  92,   6, 116,  69, -61,	  48,-123,  84,  81, 111,  56,  38, -81,	 -18,   6, 128,  72, -15,-123,  36, -36,	  59,  48, -15,  68,-127, 110, -86,  89,	  13, -37,  48,  29,   5,  75, 104, -24,	  91,-128, 105, -95,  44, 113, -94, -94,	  85,-110, -67, 111, -60,  80, -44,-104,	  81,  68, -54,  33, -96, -92,  10,  61,	 120,  -5,   0, -52,  28,  42, -64,  64,	  39,  37,  83, -82,   5, -39, -42, -83,	 128, 121, -26,  11,-107,-112,-122, -60,	-121,  91, -64,   5, 107,  88,  82,   6,	   4, -69,  -1,  70,  40,   2, -90,  57,	 -88, -54, 115,  -3, -42, -85,  84,  80,	 -69, -90,  34,-120,  43, -14,   2,  79,	 -79,  38, -69,-122, -24, -63, -49,  13,	  11,-106, -26, -61,  12, 122, -95,  56,	  17, 114, -66, -50,  90,-125,  64, -19,	 -70,   6, -59, -81, -17,  88, 118, -21,	  15, -31, -91,  22, -64, -28,  82, -11,	 -44,  64,  37,  63,  31, -14,  -5,  37,	-101,  38, -82, -53,-119,-116, -46,  82,	 106, -40,  97,-118,-104,  56,-128, -39,	 -99,   9,  83,  26,  87, 112,  71,  21,	  -7,   5,  65,-116,  68, 116, -26, 113,	  10, -39,  99, -52, -56,  92,-100,   4,	  97,  46,  49, -37,  72,-118, 103, -86,	-108,-128, -64,  19,  69, -12,  43,  31,	  58,  68,  36, -62, -98,  22,  54,  34,	 -24,-116, -52, 100, -31,  96,  22, -65,	  41, -26,  24,  79, -23,-119,  30, 120,	 117, -41, -85, -60, -85, 107,  86,  29,	 -76,-106,   0,   6,-111, -71,  68,  55,	  54,  58, -44, -75,  60, -26,  33, -25,	  77, -47, -41,-103,  80,  51,-116, 122,	  68,-109, -38, 122,  -3,  48,  64, -82,	  41, 115,  62, -14,-116,  81, 119, 121,	   5,  68, 121,  88, -18,  29, -27,  90,	-122, -98,  35, -34, -89, 112,  49,  37,	-111,  60, 126,-123,  42,-112, 115,  90,	  73,-124, -46,  86, 120,-116, 122, -16,	 127,  56,-127,  36, -83,  75,  83, -11,	 112,  45,-121, -63, -56, 115,   1,-101,	 114, -90, -49,  12, -81,-110,  32, -87,	  -6, 100, 102, -37, 122, -47,   6,  49,	  75, -56,  38, 105,-125,-122, 126, 102,	  13, 121,  76, -29, -55,  20,  61, -44,	 -11,  13, -50,  42,-109, -89,  37,  -4,	  34,  94,-116, -72,  18, -23,-100, 109,	 104,  64,  -7, 125,  49, -21,  86,  48,	 -61,  77,  75, -20,-101, 103, -32,  19,	 110, -28,  22,  68, -80,  93, -36, -76,	-105,-104,  61, 108, 101,  74, -10, -62,	 127, -41,  30, -91, -89,  61,  83, -28,	 120,-101,  96, 120, -56, 124,  43,  27,	  -4,  -7, 120,-114,  89, -22, -68, -14,	-107,   7, 127, 119,-108, -13,  84, -72,	-123,  34, 128, -64, -21, -23,-125, 117,	-120,  32,-112, -73,  44, 121,  51,  76,	  11, -29,-115,  -6,  39,  77, -29,  -6,	  41,  58, -11, 107, 125, -70,   9, -17,	  35,   8,  11, -95, -15, -37, -99, -94,	   2, -73, -94, -30, -15,   2, 100, 101,	   2,  78,-128,  34,  89,  28,  26,-100,	  79,  31, 107,  -7, -63,-101, -71,  69,	 -45,  66,  41, -77,-118,  42, -46,  -4,	  -1, -18,  29,-128, 104,  -9, -75,  68,	   1, -68,  48, -31,  36, -28,   3, -99,	  41,  53, -16,  22, 115, -83,  16, -94,	 -33,  19, 112, -38, -80, -24,  42,  27,	  -7,-123,  18,  28,-112, 122,  68,  34,	-123,  31,-110,  17,  39, -69,-107,  76};"

SWIFFT_M = 32
SWIFFT_N = 64

def amx_hadamard_with_collapsing_sum_prototype(dummy_fftout: np.ndarray, key: np.ndarray) -> np.ndarray:

    # transpose dummy fftout
    dummy_fftout_transpose = dummy_fftout.T

    # partition dummy_fftout_T into 4 16x32 matrices
    dummy_fftout_partitions = np.split(dummy_fftout_transpose, 4, axis=0)
    # split key int0 4 32x16 partitions
    key_partitions = np.split(key, 4, axis=1)


    # concatenate each dummy fftout partition with a 16x32 zero matrix to get 16x64
    # this is to fit the 16x64 size of the amx tile A 
    dummy_fftout_partitions_padded = [np.concatenate([partition, np.zeros((16, 32))], axis=1) for partition in dummy_fftout_partitions]
    assert(dummy_fftout_partitions_padded[0].shape == (16, 64))
    # concatenate each key partition with a 32x16 zero matrix to get 64x16
    # this is to fit the 64x16 size of the amx tile A 
    key_partitions_padded = [np.concatenate([partition, np.zeros((32, 16))], axis=0) for partition in key_partitions]
    assert(key_partitions_padded[0].shape == (64, 16))

    # compute matmul of likewise padded partions of A and B (using amx tiles in full implementation)
    likewise_AB_partitions_products = [np.matmul(dummy_fftout_partitions[i], key_partitions[i]) for i in range(4)]

    # concatenate diagonals
    return np.concatenate([np.diagonal(p) for p in likewise_AB_partitions_products])

def extract_matrix_from_text(input_text: str, num_rows: int, num_columns: int) -> np.ndarray:
    # extract numbers
    numbers = re.findall(r'-?\d+', input_text)
    # to int (skipping the 16 from int16_t)
    int_numbers = [int(num) for num in numbers][1:]
    # return as np array
    return np.array(int_numbers).reshape(num_rows, num_columns)

def generate_dummy_fftout():
    return np.random.randint(0, 100, (32, 64))

def test_algorithm_property_holds():
    """ 
    Ensures that the hadamard product of two 32x64 matrices A and B is  
    equal to the diagonal of the product of the transpose of A and B. 
    """

    # key (32x64)
    key: np.ndarray = extract_matrix_from_text(ORIGINAL_C_PI_KEY,  SWIFFT_M, SWIFFT_N)
    # dummy fftout of size 64x32
    dummy_fftout = generate_dummy_fftout()
    # transpose dummy fftout
    dummy_fftout_transpose = dummy_fftout.T

    # matmul dummy_fftout.T by key
    full_matmul_result = np.matmul(dummy_fftout_transpose, key) 
    assert full_matmul_result.shape == (64, 64)

    # perform hadamard product on dummy_fftout and key and collapsing sum along columns
    hadamard_products = np.sum(np.multiply(dummy_fftout, key), axis=0)

    # assert that the concatenated diagonals are equal to the diagonal of the result
    assert np.array_equal( np.diagonal(full_matmul_result), hadamard_products)
    print("test_algorithm_property_holds()... PASS!")

def test_partitioning_into_4_16x32_and_4_32x16_works_also():
    
    # get key (32x64)
    key: np.ndarray = extract_matrix_from_text(ORIGINAL_C_PI_KEY,  SWIFFT_M, SWIFFT_N)

    # get dummy fftout (32x64)
    dummy_fftout = generate_dummy_fftout()
    # transpose dummy fftout (64x32)
    dummy_fftout_transpose = dummy_fftout.T

    # split key int0 4 32x16 partitions
    key_partitions = np.split(key, 4, axis=1)
    # partition dummy_fftout_T into 4 16x32 matrices
    dummy_fftout_partitions = np.split(dummy_fftout_transpose, 4, axis=0)

    # compute likewise partioned matmul of A and B
    hadamard_products = [np.matmul(dummy_fftout_partitions[i], key_partitions[i]) for i in range(4)]

    # concatenate diagonals
    concatenated_diagonals = np.concatenate([np.diagonal(hadamard_product) for hadamard_product in hadamard_products])
    
    # ensure concatenated diagonals same 
    expected = np.diag(np.matmul(dummy_fftout_transpose, key))

    # assert that the concatenated diagonals are equal to the diagonal of the result
    assert np.array_equal(concatenated_diagonals, expected)

    print("test_partitioning_into_4_16x32_and_4_32x16_works_also()... PASS!")


def test_partitioning_into_4_16x32_and_4_32x16_with_padding_works_also():

    # get key (32x64)
    key: np.ndarray = extract_matrix_from_text(ORIGINAL_C_PI_KEY,  SWIFFT_M, SWIFFT_N)

    # get dummy fftout (32x64)
    dummy_fftout = generate_dummy_fftout()

    # perform full algoritm (which includes padding to amx tile size)
    concatenated_diagonals = amx_hadamard_with_collapsing_sum_prototype(dummy_fftout, key)

    # ensure concatenated diagonals same
    expected = np.diag(np.matmul(dummy_fftout.T, key))  
    # assert that the concatenated diagonals are equal to the diagonal of the result
    assert np.array_equal(concatenated_diagonals, expected)

    print("test_partitioning_into_4_16x32_and_4_32x16_with_padding_works_also()... PASS!")


if __name__ == '__main__':
    test_algorithm_property_holds()
    test_partitioning_into_4_16x32_and_4_32x16_works_also()
    test_partitioning_into_4_16x32_and_4_32x16_with_padding_works_also()