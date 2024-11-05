#include <assert.h>
#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/syscall.h>
#include <stdbool.h>
#include <time.h>   
#include <stdlib.h>

#define ARCH_GET_XCOMP_PERM     0x1022
#define ARCH_REQ_XCOMP_PERM     0x1023
#define XFEATURE_XTILECFG       17
#define XFEATURE_XTILEDATA      18

#define MAX 1024 

/* Set_tiledata_use() - Invoke syscall to set ARCH_SET_STATE_USE */
static bool set_tiledata_use()
{
   if (syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)) 
   {
      printf("\n Fail to do XFEATURE_XTILEDATA \n\n");
      return false;
   }
   else
   {
      printf("\n TILE DATA USE SET - OK \n\n");
      return true;
   }

   return true;
}

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

/*
  AMX's implementation of dpbuud.
   - A = M x 4K
   - B = K x 4N
   - C = M x N (but is uint32_t)
*/
inline void amx_dpbuud(int M, int K, int N, uint8_t *A, uint8_t *B, uint32_t *C) {
  unsigned char config[] = {
      0x01,                                      // ID
      0x00,                                      // start row
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // reserved
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // reserved
      4 * K, 0x00,                               // bytes per row tile 0
      4 * N, 0x00,                               // bytes per row tile 1
      4 * N, 0x00,                               // bytes per row tile 2
      0x01, 0x00,                                // bytes per row tile 3
      0x00, 0x00,                                // bytes per row tile 4
      0x00, 0x00,                                // bytes per row tile 5
      0x00, 0x00,                                // bytes per row tile 6
      0x00, 0x00,                                // bytes per row tile 7
      0x00, 0x00,                                // bytes per row tile 8
      0x00, 0x00,                                // bytes per row tile 9
      0x00, 0x00,                                // bytes per row tile 10
      0x00, 0x00,                                // bytes per row tile 11
      0x00, 0x00,                                // bytes per row tile 12
      0x00, 0x00,                                // bytes per row tile 13
      0x00, 0x00,                                // bytes per row tile 14
      0x00, 0x00,                                // bytes per row tile 15
      M,                                         // rows tile 0
      K,                                         // rows tile 1
      M,                                         // rows tile 2
      0x01,                                      // rows tile 3
      0x00,                                      // rows tile 4
      0x00,                                      // rows tile 5
      0x00,                                      // rows tile 6
      0x00,                                      // rows tile 7
      0x00,                                      // rows tile 8
      0x00,                                      // rows tile 9
      0x00,                                      // rows tile 10
      0x00,                                      // rows tile 11
      0x00,                                      // rows tile 12
      0x00,                                      // rows tile 13
      0x00,                                      // rows tile 14
      0x00                                       // rows tile 15
  };

  _tile_loadconfig(config);

  _tile_zero(2);
  _tile_loadd(0, A, 4 * K);
  _tile_loadd(1, B, 4 * N);

  _tile_dpbuud(2, 0, 1);

  _tile_stored(2, C, 4 * N);
}

/*
  Takes a matrix old_B, and converts it from its 4M x N representation
  to a M x 4N representation, which matches AMX's tile format.
   - old_B = 4M x N,
   - new_B = M x 4N,
*/
inline void transform(int M, int N, uint8_t *new_B, uint8_t *old_B) {
  for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
      for (int m_in = 0; m_in < 4; m_in++) {
        new_B[m * 4 * N + (4 * n + m_in)] = old_B[(4 * m + m_in) * N + n];
      }
    }
  }
}

/*
  Makes use of AMX's tile instruction to perform matmul on uint8_t matrices.
  Requires a memory transform prior to loading data into the tile.
   - A = M x 4K
   - B = 4K x N
   - C = M x N (but is uint32_t)
*/
inline void amx_matmul(int M, int K, int N, uint8_t *A, uint8_t *B, uint32_t *C) {
  uint8_t new_B[K][4 * N];
  transform(K, N, new_B, B);
  amx_dpbuud(M, K, N, A, new_B, C);
}


/**
 * For the purposes of accelerating LibSWIFFT, we are working under Z_257. This means that to 
 * represent the full range of values 0-256 (inclusive), we need at least 9 bits of precision.
 * 
 * This function splits a uint16_t array into two separate uint8_t arrays, one containing the 
 * low 8 bits and the other containing the high 1 bit of each element. For which these arrays 
 * can be operated on seperately, and later rejoined into a single integer.
 */
inline void bit_split(const uint16_t *input, uint8_t *low_bits, uint8_t *high_bits, size_t length) {
    for (size_t i = 0; i < length; ++i) {

        // Extract the low 8 bits (bits 0-7)
        low_bits[i] = input[i] & 0xFF; // 0xFF = 11111111 in binary
        
        // Extract the high 1 bits (bit 9)
        high_bits[i] = (input[i] >> 8) & 0x01; // or 0x01 potentially
    }
}

/* Recombines two uint8_t arrays into a single uint32_t array. Inverse of bit_split() */
inline void bit_recombine(const uint32_t *low_bits, const uint32_t *high_bits, uint32_t *output, size_t length) {
    for (size_t i = 0; i < length; ++i) {
        output[i] = (high_bits[i] << 8) | low_bits[i];
    }
}

/* left shift uint32_t buffer using naive method */
inline void naive_left_shift_buffer32(uint32_t *buf, size_t length, int shift) {
    for (size_t i = 0; i < length; ++i) {
        buf[i] <<= shift;
    }
}

/* left shift uint32_t buffer using AVX-512 */
inline void left_shift_buffer32_avx512(uint32_t *buf, size_t length, int shift) {
    size_t i = 0;
    size_t simd_width = 16; // AVX-512 can process 16 uint32_t elements at a time

    // Load the shift amount into an AVX-512 register
    __m512i shift_vec = _mm512_set1_epi32(shift);

    // Process buffer in chunks of 16 
    for (; i + simd_width <= length; i += simd_width) {
        __m512i data = _mm512_loadu_si512((__m512i*)&buf[i]); // load 16 elements into AVX-512 register
        __m512i result = _mm512_sllv_epi32(data, shift_vec);  // left shift them        
        _mm512_storeu_si512((__m512i*)&buf[i], result);       // store the result back
    }

    // handle remaining elements
    for (; i < length; ++i) buf[i] <<= shift;
}

/* add two uint32_t buffers */
inline void naive_add_buffer32(uint32_t *buf1, uint32_t *buf2, uint32_t *res, size_t length) {
    for (size_t i = 0; i < length; ++i) {
        res[i] = buf1[i] + buf2[i];
    }
} 

/* Add two uint32_t buffers using AVX-512 */
inline void add_buffer32_avx512(uint32_t *buf1, uint32_t *buf2, uint32_t *res, size_t length) {
    size_t i = 0;
    size_t simd_width = 16; // Number of uint32_t elements per AVX-512 vector

    // Process as many full SIMD-width blocks as possible
    for (; i + simd_width <= length; i += simd_width) {
        __m512i vec1 = _mm512_loadu_si512((__m512i const*)&buf1[i]); // Load 16 elements from buf1
        __m512i vec2 = _mm512_loadu_si512((__m512i const*)&buf2[i]); // Load 16 elements from buf2
        __m512i sum = _mm512_add_epi32(vec1, vec2);                  // Perform element-wise addition
        _mm512_storeu_si512((__m512i*)&res[i], sum);                 // Store the result back to the res buffer
    }

    // Handle any remaining elements that don't fit into a full SIMD vector
    for (; i < length; ++i) {
        res[i] = buf1[i] + buf2[i];
    }
}

/* uint16_t matmul w/ bit splitting to uint8_t */
static inline void bit_split_amx_matmul(const uint16_t* src1, const uint16_t* src2, uint32_t* res){

    // mem for split buffers
    uint8_t src1_hi[MAX]; // A
    uint8_t src1_lo[MAX]; // B
    uint8_t src2_hi[MAX]; // C
    uint8_t src2_lo[MAX]; // D

    // Split into low and high bits
    bit_split(src1, src1_lo, src1_hi, MAX);
    bit_split(src2, src2_lo, src2_hi, MAX);

    // init mem for products
    uint32_t AC[MAX/4]  __attribute__((aligned(64)));
    uint32_t AD[MAX/4]  __attribute__((aligned(64)));
    uint32_t BC[MAX/4]  __attribute__((aligned(64)));
    uint32_t BD[MAX/4]  __attribute__((aligned(64)));
    memset(AC, 0, sizeof(AC));
    memset(AD, 0, sizeof(AD));
    memset(BC, 0, sizeof(BC));
    memset(BD, 0, sizeof(BD));

    // compute AC, AD, BC, BD
    amx_matmul(16, 16, 16, src1_hi, src2_hi, AC); // A * C
    amx_matmul(16, 16, 16, src1_hi, src2_lo, AD); // A * D
    amx_matmul(16, 16, 16, src1_lo, src2_hi, BC); // B * C
    amx_matmul(16, 16, 16, src1_lo, src2_lo, BD); // B * D

    // left shift w/ AVX-512  (picked up from FOIL-ing)
    left_shift_buffer32_avx512(AC, MAX/4, 16); // AC * 256 * 256
    left_shift_buffer32_avx512(BC, MAX/4, 8); // BD * 256
    left_shift_buffer32_avx512(AD, MAX/4, 8); // AD * 256
    
    // recombine w/ AVX-512
    add_buffer32_avx512(AC, BD, res, MAX/4);
    add_buffer32_avx512(res, BC, res, MAX/4);
    add_buffer32_avx512(res, AD, res, MAX/4);
}

/* ensures values are within 0-256 inclusive */
inline uint16_t ensure_correct_range(uint16_t* buf, size_t length) {
    for (size_t i = 0; i < length; ++i) {
        if (buf[i] > 256) {
            return 0; // or some error code
        }
    }
    return 1; // success
}

/* Print uint32_t buffer */
inline void print_buffer32(uint32_t* buf, uint32_t rows, uint32_t colsb)
{
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
inline void init_random_buffer16(uint16_t *buf, uint32_t size) {
  for (uint32_t i = 0; i < size; i++){
    buf[i] = rand() % 257; 
  }
}

/* matrix w/ sequentially increasing values within Z_257 */
const uint16_t A[1024] = {
  0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 
  64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 
  128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 
  192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 
  256, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 
  63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 
  127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 
  191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 
  255, 256, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 
  62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 
  126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 
  190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 
  254, 255, 256, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 
  61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 
  125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 
  189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 
};

void test_const_input(bool verbose) {
    
    // ensure input values are within Z_257
    if (ensure_correct_range(A, MAX)) {
        if (verbose) printf("\nAll input values are within the range of Z_257...\n");
    } else {
        if (verbose) printf("\nSome values are out of range!\n");
    }

    // niave result array
    uint32_t naive_res[256];
    memset(naive_res, 0, sizeof(naive_res));

    // naive matmul
    naive_matmul_uint16_t(A, A, naive_res, 16, 64, 16);
    
    // bit split matmul result array
    uint32_t bit_split_result[256];
    memset(bit_split_result, 0, sizeof(bit_split_result));

    // bit split naive matmul
    bit_split_amx_matmul(A, A, bit_split_result);


    // print result 
    if (verbose) {
        printf("\nNaive Result:\n");
        print_buffer32(naive_res, 16, 16);   
    
        printf("\nBit Split Result:\n");
        print_buffer32(bit_split_result, 16, 16);
    }
    // ensure naive and naive w/ bit split are equal
    for (size_t i = 0; i < 256; ++i) {
        assert(naive_res[i] == bit_split_result[i]);
    }

    printf("\nResults match for static inputs!\n");
}

/* Test on fuzz input */
void test_fuzz_input (int rounds, bool verbose) {

    for (int i=0; i<rounds; i++){

        // initialize random input
        uint16_t fuzz_lhs[MAX];
        uint16_t fuzz_rhs[MAX];
        init_random_buffer16(fuzz_lhs, MAX);
        init_random_buffer16(fuzz_rhs, MAX);

        // ensure input values are within Z_257
        if (ensure_correct_range(A, MAX)) {
            if (verbose) printf("\nAll input values are within the range of Z_257...\n");
        } else {
            if (verbose) printf("\nSome values are out of range!\n");
        }

        // naive result array
        uint32_t naive_res[256];
        memset(naive_res, 0, sizeof(naive_res));

        // naive matmul
        naive_matmul_uint16_t(fuzz_lhs, fuzz_rhs, naive_res, 16, 64, 16);
        
        // bit split matmul result array
        uint32_t bit_split_result[256];
        memset(bit_split_result, 0, sizeof(bit_split_result));

        // bit split naive matmul
        // bit_split_naive_matmul(fuzz_lhs, fuzz_rhs, bit_split_result);
        bit_split_amx_matmul(fuzz_lhs, fuzz_rhs, bit_split_result);

        // print result 
        if (verbose) {
            printf("\nNaive Result:\n");
            print_buffer32(naive_res, 16, 16);   
        
            printf("\nBit Split Result:\n");
            print_buffer32(bit_split_result, 16, 16);
        }
        // ensure naive and naive w/ bit split are equal
        for (size_t i = 0; i < 256; ++i) {
            if (naive_res[i] != bit_split_result[i]) {
                printf("Mismatch at index %zu: naive_res[%zu] = %u, bit_split_result[%zu] = %u\n", i, i, naive_res[i], i, bit_split_result[i]);
            }
            assert(naive_res[i] == bit_split_result[i]);
        }
    }
    printf("\nResults match for fuzz inputs!\n\n\n");
}

int main(void){

    // ask permission to use tile data
    set_tiledata_use();

    test_const_input(false);
    test_fuzz_input(100, false);

    return 0;
}