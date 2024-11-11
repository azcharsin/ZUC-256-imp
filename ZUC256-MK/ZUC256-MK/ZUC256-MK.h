#ifndef _ZUC256_AVX512_H_H_
#define _ZUC256_AVX512_H_H_

#include <immintrin.h>
typedef unsigned char u8;
typedef unsigned int u32;

void ZUC256_AVX512(u32* ks, int wordlen, const u8* k, const u8* iv);

void ZUC256_CRYPT_AVX512(u32* C, const u8* CK, const u8* IV, const u32* M, int LENGTH);


void ZUC256_MAC_AVX512(u32* MAC, int MAC_BITLEN, const u8* IK, const u8* IV, const u32* M, const u32 LENGTH);

#endif
