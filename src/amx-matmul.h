#include <assert.h>
#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/syscall.h>
#include <stdbool.h>

#define ARCH_GET_XCOMP_PERM     0x1022
#define ARCH_REQ_XCOMP_PERM     0x1023
#define XFEATURE_XTILECFG       17
#define XFEATURE_XTILEDATA      18

// max buffer size for ANX tile in bytes
#define MAX 1024 

/* Set_tiledata_use() - Invoke syscall to set ARCH_SET_STATE_USE */
inline bool set_tiledata_use(){
   if (syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)) {
      printf("\n Fail to do XFEATURE_XTILEDATA \n\n");
      return false;
   }
   else{
      printf("\n TILE DATA USE SET - OK \n\n");
      return true;
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
    for (size_t i = 0; i < length; ++i) buf[i] <<= shift;
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
    for (size_t i = 0; i < length; ++i) res[i] = buf1[i] + buf2[i];
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
    for (; i < length; ++i) res[i] = buf1[i] + buf2[i];   
}

/* 
  uint16_t matmul w/ bit splitting to uint8_t 

   - src1 = M x 4K
   - src2 = 4K x N
   - res = M x N (but is uint32_t)
*/
inline void bit_split_amx_matmul( const uint16_t* src1, const uint16_t* src2, uint32_t* res, int M, int K, int N ){

    const int src1_elements = M * 4 * K;
    const int src2_elements = 4 * K * N;  
    const int output_elements = M * N;

    // mem for split buffers
    uint8_t src1_hi[src1_elements]; // A
    uint8_t src1_lo[src1_elements]; // B
    uint8_t src2_hi[src2_elements]; // C
    uint8_t src2_lo[src2_elements]; // D

    // Split into low and high bits
    bit_split(src1, src1_lo, src1_hi, src1_elements);
    bit_split(src2, src2_lo, src2_hi, src2_elements);

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
    amx_matmul(M, K, N, src1_hi, src2_hi, AC); // A * C
    amx_matmul(M, K, N, src1_hi, src2_lo, AD); // A * D
    amx_matmul(M, K, N, src1_lo, src2_hi, BC); // B * C
    amx_matmul(M, K, N, src1_lo, src2_lo, BD); // B * D

    // left shift w/ AVX-512  (picked up from FOIL-ing)
    left_shift_buffer32_avx512(AC, output_elements, 16); // AC * 256 * 256
    left_shift_buffer32_avx512(BC, output_elements, 8); // BD * 256
    left_shift_buffer32_avx512(AD, output_elements, 8); // AD * 256
    
    // recombine w/ AVX-512
    add_buffer32_avx512(AC, BD, res, output_elements);
    add_buffer32_avx512(res, BC, res, output_elements);
    add_buffer32_avx512(res, AD, res, output_elements);
}