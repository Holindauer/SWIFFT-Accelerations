#include <stdio.h>
#include <string.h>

#include "amx-fftsum.h"

/* naive int16_t matrix transposition */
static void naive_transpose_matrix_int16_t(const int16_t *input, int16_t *output, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            output[j * rows + i] = input[i * cols + j];
        }
    }
}

void test_transpose_32x64_sse2() {
    
    // init input mats
    int16_t naive_input[SWIFFT_M*SWIFFT_N] __attribute__((aligned(16)));
    int16_t vectorized_input[SWIFFT_M*SWIFFT_N] __attribute__((aligned(16)));
    
    for (int i = 0; i < 32 * 64; i++) {
        naive_input[i] = i % 256;
        vectorized_input[i] = i % 256;
    }

    // init output mats
    int16_t output_naive[32 * 64] __attribute__((aligned(16)));
    int16_t output_vectorized[32 * 64] __attribute__((aligned(16)));
    memset(output_naive, 0, sizeof(output_naive));
    memset(output_vectorized, 0, sizeof(output_vectorized));

    // transpose using naive method
    naive_transpose_matrix_int16_t(naive_input, output_naive, 32, 64);

    // transpose using sse2
    transpose_32x64_sse2(vectorized_input, output_vectorized, 32, 64);

    // compare results
    for (int i = 0; i < 32 * 64; i++) {
        if (output_naive[i] != output_vectorized[i]) {
            puts("test_transpose_32x64_sse2... FAIL!");
            return;
        }
    }

    puts("test_transpose_32x64_sse2... PASS!");
}

int main(void){

    test_transpose_32x64_sse2();

    return 0;
}