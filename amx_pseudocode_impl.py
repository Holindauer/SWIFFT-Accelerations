import numpy as np  
from create_PI_key_partition_lookups import create_padded_partitions

"""
This script implements the pseudocode for how the TMUL unit performs matrix multiplication
under the hood. That psuedocode is as follows:

Synopsis

void __tile_dpbssd (__tile1024i* dst, __tile1024i src0, __tile1024i src1)
#include <immintrin.h>
Instruction: tdpbssd tmm, tmm, tmm
CPUID Flags: AMX-INT8

Description
Compute dot-product of bytes in tiles with a source/destination accumulator. Multiply groups
of 4 adjacent pairs of signed 8-bit integers in src0 with corresponding signed 8-bit integers 
in src1, producing 4 intermediate 32-bit results. Sum these 4 results with the corresponding 
32-bit integer in dst, and store the 32-bit result back to tile dst. The shape of tile is 
specified in the struct of __tile1024i. The register of the tile is allocated by compiler.

Operation
DEFINE DPBD(c, x, y) {
	tmp1 := SignExtend32(x.byte[0]) * SignExtend32(y.byte[0])
	tmp2 := SignExtend32(x.byte[1]) * SignExtend32(y.byte[1])
	tmp3 := SignExtend32(x.byte[2]) * SignExtend32(y.byte[2])
	tmp4 := SignExtend32(x.byte[3]) * SignExtend32(y.byte[3])
	RETURN c + tmp1 + tmp2 + tmp3 + tmp4
}
FOR m := 0 TO dst.rows - 1
	tmp := dst.row[m]
	FOR k := 0 TO (src0.colsb / 4) - 1
		FOR n := 0 TO (dst.colsb / 4) - 1
			tmp.dword[n] := DPBD(tmp.dword[n], src0.row[m].dword[k], src1.row[k].dword[n])
		ENDFOR
	ENDFOR
	write_row_and_zero(dst, m, tmp, dst.colsb)
ENDFOR
zero_upper_rows(dst, dst.rows)


"""

def DPBD(c, x, y) -> np.int32:
    tmp1 = np.int32(x[0]) * np.int32(y[0])
    tmp2 = np.int32(x[1]) * np.int32(y[1])
    tmp3 = np.int32(x[2]) * np.int32(y[2])
    tmp4 = np.int32(x[3]) * np.int32(y[3])
    return c + tmp1 + tmp2 + tmp3 + tmp4

def write_row_and_zero(dst, m, tmp):
    # Write the computed values to row m of dst directly
    dst[m, :] = tmp[:]

def zero_upper_rows(dst, rows):
    # Zero out any remaining rows in dst
    dst[rows:, :] = 0

def __tile_dpbssd(dst, src0, src1):
    for m in range(dst.shape[0]):
        tmp = np.copy(dst[m])  # copy to not overwrite the original 
        
        for k in range(0, src0.shape[1], 4):  # Iterate in steps of 4
            for n in range(dst.shape[1]):  # Iterate through all columns
                x = src0[m, k:k+4]  # Extract 4 elements from src0
                y = src1[k:k+4, n]  # Extract 4 elements from src1
                tmp[n] = DPBD(tmp[n], x, y)
        
        write_row_and_zero(dst, m, tmp)
    
    zero_upper_rows(dst, dst.shape[0])

    return dst



if __name__ == "__main__":
        
    # ones matrix of shape (16, 64)
    src1 = np.ones((16, 64), dtype=np.int8)

    # set src2 to first (64, 16) padded partition of SWIFFT_PI_key
    padded_partitions, _ = create_padded_partitions()
    src2 = padded_partitions[0]
    assert(src2.shape == (64, 16))

    dst = np.zeros((16, 16), dtype=np.int32)

    __tile_dpbssd(dst, src1, src2)

    print(dst)