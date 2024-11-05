#include <time.h>   
#include <stdlib.h>

#include "amx-matmul.h"

/* Naive matmul uint16_t matmul w/ zero extension to uint32_t */
static void naive_matmul_uint16_t(const uint16_t *A, const uint16_t *B, uint32_t *c, int A_rows, int A_cols, int B_cols){
  for (int i = 0; i < A_rows; i++) {
    for (int j = 0; j < B_cols; j++) {
      for (int k = 0; k < A_cols; k++){
        // zero extend mul args from 16->32
        c[i * B_cols + j] += (uint32_t)A[i * A_cols + k] * (uint32_t)B[k * B_cols + j];
      }
    }
  }
} 

/* Print uint32_t buffer */
static void print_buffer32(uint32_t* buf, uint32_t rows, uint32_t colsb){
   for (int i = 0; i < rows; i++) {
     for (int j = 0; j < (colsb); j++)
     {
         printf("%u ", buf[i * colsb + j]);
     }
     printf("\n");
   }
   printf("\n");
}

/* Initialize random uint16_t buffer within Z_257 range */
static void init_random_buffer16(uint16_t *buf, uint32_t size) {
  for (uint32_t i = 0; i < size; i++) buf[i] = rand() % 257; 
}

/* Test amx accuracy on fuzz input */
static void test_amx_accuracy_fuzz (int rounds, bool verbose) {
    for (int i=0; i<rounds; i++){

        // init random input
        uint16_t fuzz_lhs[MAX], fuzz_rhs[MAX];
        init_random_buffer16(fuzz_lhs, MAX);
        init_random_buffer16(fuzz_rhs, MAX);

        // result buffers
        uint32_t naive_res[256], bit_split_result[256];
        memset(naive_res, 0, sizeof(naive_res));
        memset(bit_split_result, 0, sizeof(bit_split_result));

        // perform both matmuls
        naive_matmul_uint16_t(fuzz_lhs, fuzz_rhs, naive_res, 16, 64, 16);
        bit_split_amx_matmul(fuzz_lhs, fuzz_rhs, bit_split_result);

        // print result 
        if (verbose) {
            printf("\nNaive Result:\n");
            print_buffer32(naive_res, 16, 16);   
            printf("\nBit Split Result:\n");
            print_buffer32(bit_split_result, 16, 16);
        }
        // ensure naive and naive w/ bit split are equal
        for (size_t i = 0; i < 256; ++i)
            assert(naive_res[i] == bit_split_result[i]);
    
    }
    puts("test_amx_accuracy_fuzz... PASS!!\n");
}

/* time trial for AMX mamtul against naive */
static void test_amx_time_fuzz(int rounds) {
    clock_t start, end;
    double amx_cpu_time, naive_cpu_time;

    // init random input
    uint16_t fuzz_lhs[MAX], fuzz_rhs[MAX];
    init_random_buffer16(fuzz_lhs, MAX);
    init_random_buffer16(fuzz_rhs, MAX);

    // result buffers
    uint32_t naive_res[256], bit_split_result[256];
    memset(naive_res, 0, sizeof(naive_res));
    memset(bit_split_result, 0, sizeof(bit_split_result));

    // perform rounds number of matmuls and time them
    start = clock();
    for (int i = 0; i < rounds; i++) {
        naive_matmul_uint16_t(fuzz_lhs, fuzz_rhs, naive_res, 16, 64, 16);
    }
    end = clock();
    naive_cpu_time = ((double) (end - start)) / CLOCKS_PER_SEC;

    // perform rounds number of matmuls and time them
    start = clock();
    for (int i = 0; i < rounds; i++) {
        bit_split_amx_matmul(fuzz_lhs, fuzz_rhs, bit_split_result);
    }
    end = clock();
    amx_cpu_time = ((double) (end - start)) / CLOCKS_PER_SEC;

    // print results
    printf("Naive CPU Time: %f\n", naive_cpu_time);
    printf("AMX CPU Time: %f\n", amx_cpu_time);
    printf("AMX is %f times faster than naive\n", naive_cpu_time / amx_cpu_time);
}   

int main(void){

    // ask permission to use tile data
    set_tiledata_use();

    test_amx_accuracy_fuzz(10000, false);
    test_amx_time_fuzz(10000);

    puts("");
    return 0;   
}