#pragma once

#include <stdint.h>
#include <stdio.h>

#define SWIFFT_M 32
#define SWIFFT_N 64


void amx_fftsum(const uint16_t fftout, int m, uint16_t out);


// Dummy uint16_t PI key, // TODO replace later w/ actual
extern const uint16_t PI_key_dummy[SWIFFT_M*SWIFFT_N];