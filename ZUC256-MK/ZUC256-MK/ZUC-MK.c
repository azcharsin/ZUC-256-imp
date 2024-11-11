#include "ZUC256-MK.h"
#include <stdio.h>
#include <time.h>


#define V4ROTL(a,imm) _mm512_or_si512(_mm512_slli_epi32(a,imm),_mm512_srli_epi32(a,32-imm))		// ZMM32bit元素循环左移
#define V4SCATTER(C, vindex, v) _mm512_i32scatter_epi32(C, vindex, v, 4)	// 将寄存器v的512bit存入C
#define V4BLENDB(a, b) _mm512_mask_blend_epi8(0x5555555555555555, a, b)		// 将ab的内容混合
#define V4BLENDS(a, b) _mm512_mask_blend_epi16(0x55555555, a, b)			// 将ab的内容混合


typedef struct {
	__m512i LFSR_S[16];		// 16个512 ZMM
	__m512i F_R[2];
	__m512i BRC_X[4];
} ZUC256_AVX512_State;

__m512i p1_mask;			// 512 ZMM 
__m512i p2_mask;
__m512i p3_mask;

#define LOWER_NIBBLE_MASK 0x0F		// 低4bit
#define LOWER_5BITS_MASK 0x1F		// 低5bit
#define HIGHER_3BITS_MASK 0xE0		// 高3bit
__m512i lower_nibble_mask;
__m512i lower_5bits_mask;
__m512i higher_3bits_mask;

#define RIGHT_1BIT_MASK 0x55		
#define LEFT_1BIT_MASK 0xAA
#define RIGHT_2BITS_MASK 0x33
#define LEFT_2BITS_MASK 0xCC
#define RIGHT_4BITS_MASK 0x0F
#define LEFT_4BITS_MASK 0xF0
#define RIGHT_8BITS_MASK 0x00FF
#define LEFT_8BITS_MASK 0xFF00
__m512i right_1bit_mask;
__m512i left_1bit_mask;
__m512i right_2bits_mask;
__m512i left_2bits_mask;
__m512i right_4bits_mask;
__m512i left_4bits_mask;
__m512i right_8bits_mask;
__m512i left_8bits_mask;

__m512i k_mul_mask1;
__m512i k_mul_mask2;
__m512i t_mul_mask1;
__m512i t_mul_mask2;
__m512i shuffle_mask;
//__m128i aes_const_key;

__m512i aes_const_key;


#define MBP_MASK 0x7FFFFFFF
__m512i mbp_mask;

int SetupSign = 0;

inline void ZUC256_Setup_AVX512()
{
#define P1_MASK_128 0x09030507,0x0C000400,0x0A020F0F,0x0E000F09		// P1的128bit定义
#define P2_MASK_128 0x0209030F,0x0A0E010B,0x040C0007,0x05060D08		// P2的128bit定义
#define P3_MASK_128 0x0D0C0900,0x050D0303,0x0F0A0D00,0x060A0602		// P3的128bit定义
#define P1_MASK P1_MASK_128,P1_MASK_128,P1_MASK_128,P1_MASK_128		// P1的512bit定义，用于适应寄存器
#define P2_MASK P2_MASK_128,P2_MASK_128,P2_MASK_128,P2_MASK_128		// P2的512bit定义，用于适应寄存器
#define P3_MASK P3_MASK_128,P3_MASK_128,P3_MASK_128,P3_MASK_128		// P3的512bit定义，用于适应寄存器

#define K_MUL_MASK1_128 0xD3D20A0B,0xB8B96160,0xB3B26A6B,0xD8D90100
#define K_MUL_MASK2_128 0x29AB63E1,0xEE6CA426,0x0F8D45C7,0xC84A8200
#define K_MUL_MASK1 K_MUL_MASK1_128,K_MUL_MASK1_128,K_MUL_MASK1_128,K_MUL_MASK1_128
#define K_MUL_MASK2 K_MUL_MASK2_128,K_MUL_MASK2_128,K_MUL_MASK2_128,K_MUL_MASK2_128

#define T_MUL_MASK1_128 0x5B867FA2,0xA479805D,0x538E77AA,0xAC718855
#define T_MUL_MASK2_128 0x47DE73EA,0x33AA079E,0xD940ED74,0xAD349900
#define T_MUL_MASK1 T_MUL_MASK1_128,T_MUL_MASK1_128,T_MUL_MASK1_128,T_MUL_MASK1_128
#define T_MUL_MASK2 T_MUL_MASK2_128,T_MUL_MASK2_128,T_MUL_MASK2_128,T_MUL_MASK2_128

#define AES_SHUF_MASK_128 0x0306090c,0x0f020508,0x0b0e0104,0x070a0d00
#define AES_SHUF_MASK  AES_SHUF_MASK_128,AES_SHUF_MASK_128,AES_SHUF_MASK_128,AES_SHUF_MASK_128
#define AES_CONST_KEY  0x63

	if (SetupSign == 1) return;

	SetupSign = 1;

	p1_mask = _mm512_set_epi32(P1_MASK);		// P1存入寄存器
	p2_mask = _mm512_set_epi32(P2_MASK);
	p3_mask = _mm512_set_epi32(P3_MASK);

	lower_nibble_mask = _mm512_set1_epi8(LOWER_NIBBLE_MASK);		// 将64*0x0F存入寄存器
	lower_5bits_mask = _mm512_set1_epi8(LOWER_5BITS_MASK);		// 将64*0x1F存入寄存器
	higher_3bits_mask = _mm512_set1_epi8(HIGHER_3BITS_MASK);		// 将64*0xE0存入寄存器

	right_1bit_mask = _mm512_set1_epi8(RIGHT_1BIT_MASK);			// 将64*0x55存入寄存器
	left_1bit_mask = _mm512_set1_epi8(LEFT_1BIT_MASK);			// 将64*0xAA存入寄存器
	right_2bits_mask = _mm512_set1_epi8(RIGHT_2BITS_MASK);		// 将64*0x33存入寄存器
	left_2bits_mask = _mm512_set1_epi8(LEFT_2BITS_MASK);			// 将64*0xCC存入寄存器
	right_4bits_mask = _mm512_set1_epi8(RIGHT_4BITS_MASK);		// 将64*0x0F存入寄存器
	left_4bits_mask = _mm512_set1_epi8(LEFT_4BITS_MASK);			// 将64*0xF0存入寄存器
	right_8bits_mask = _mm512_set1_epi16(RIGHT_8BITS_MASK);		// 将32*0x00FF存入寄存器
	left_8bits_mask = _mm512_set1_epi16(LEFT_8BITS_MASK);			// 将32*0xFF00存入寄存器

	k_mul_mask1 = _mm512_set_epi32(K_MUL_MASK1);					// K1存入
	k_mul_mask2 = _mm512_set_epi32(K_MUL_MASK2);					// K2存入
	t_mul_mask1 = _mm512_set_epi32(T_MUL_MASK1);					// T1
	t_mul_mask2 = _mm512_set_epi32(T_MUL_MASK2);					// T2
	shuffle_mask = _mm512_set_epi32(AES_SHUF_MASK);				// AES的行移位
	//aes_const_key = V1SET1B(AES_CONST_KEY);				// AES的轮密钥

	aes_const_key = _mm512_set1_epi8(AES_CONST_KEY);

	mbp_mask = _mm512_set1_epi32(MBP_MASK);						// 不知道是什么
}

///////////////////////////////////////////////////////////////////////////////////////////////////////

inline __m512i sbox0(const __m512i in)		// S0 
{
	__m512i hi = _mm512_and_si512(_mm512_srli_epi32(in, 4), lower_nibble_mask);	//	取出每8bit的高4位
	__m512i low = _mm512_and_si512(in, lower_nibble_mask);				//	取出每8bit的低4位

	const __m512i t1 = _mm512_xor_si512(hi, _mm512_shuffle_epi8(p1_mask, low));
	const __m512i t2 = _mm512_xor_si512(low, _mm512_shuffle_epi8(p2_mask, t1));
	const __m512i t3 = _mm512_xor_si512(t1, _mm512_shuffle_epi8(p3_mask, t2));

	const __m512i out = _mm512_or_si512(t2, _mm512_slli_epi32(t3, 4));

	low = _mm512_and_si512(_mm512_srli_epi32(out, 3), lower_5bits_mask);
	hi = _mm512_and_si512(_mm512_slli_epi32(out, 5), higher_3bits_mask);

	return _mm512_or_si512(hi, low);
}


///////////////////////////////////////////////////////////////////////////////////////////////////////

inline __m512i sbox1(const __m512i in)		// S1
{
	__m512i low = _mm512_shuffle_epi8(k_mul_mask1, _mm512_and_si512(in, lower_nibble_mask));
	__m512i hi = _mm512_shuffle_epi8(k_mul_mask2, _mm512_and_si512(_mm512_srli_epi32(in, 4), lower_nibble_mask));
	__m512i y_inv = _mm512_shuffle_epi8(_mm512_xor_si512(low, hi), shuffle_mask);

	y_inv = _mm512_aesenclast_epi128(y_inv, aes_const_key);

	low = _mm512_shuffle_epi8(t_mul_mask1, _mm512_and_si512(y_inv, lower_nibble_mask));
	hi = _mm512_shuffle_epi8(t_mul_mask2, _mm512_and_si512(_mm512_srli_epi32(y_inv, 4), lower_nibble_mask));

	return _mm512_xor_si512(low, hi);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////

#define	MBP2(x, k) _mm512_and_si512(_mm512_or_si512(_mm512_slli_epi32(x, k), _mm512_srli_epi32(x, 31-k)), mbp_mask)		// 31bit
#define	AMR(c) _mm512_add_epi32(_mm512_and_si512(c, mbp_mask), _mm512_srli_epi32(c, 31))

#define V_FEEDBACK(f, v)\
	v = MBP2(LFSR_S[ buf],  8); f = _mm512_add_epi32(LFSR_S[ buf], v);f = AMR(f);\
	v = MBP2(LFSR_S[ (4+buf)&0xF], 20); f = _mm512_add_epi32(f, v);f = AMR(f);\
	v = MBP2(LFSR_S[(10+buf)&0xF], 21); f = _mm512_add_epi32(f, v);f = AMR(f);\
	v = MBP2(LFSR_S[(13+buf)&0xF], 17); f = _mm512_add_epi32(f, v);f = AMR(f);\
	v = MBP2(LFSR_S[(15+buf)&0xF], 15); f = _mm512_add_epi32(f, v);f = AMR(f)

#define V_BITR_INIT() \
	BRC_X[0] = V4BLENDS(_mm512_slli_epi32(LFSR_S[(15+buf)&0xF], 1), LFSR_S[(14+buf)&0xF]);\
	BRC_X[1] = _mm512_or_si512(_mm512_slli_epi32(LFSR_S[(11+buf)&0xF], 16), _mm512_srli_epi32(LFSR_S[ (9+buf)&0xF], 15));\
	BRC_X[2] = _mm512_or_si512(_mm512_slli_epi32(LFSR_S[ (7+buf)&0xF], 16), _mm512_srli_epi32(LFSR_S[ (5+buf)&0xF], 15))

#define V_BITR() \
	BRC_X[0] = V4BLENDS(_mm512_slli_epi32(LFSR_S[(15+buf)&0xF], 1), LFSR_S[(14+buf)&0xF]);\
	BRC_X[1] = _mm512_or_si512(_mm512_slli_epi32(LFSR_S[(11+buf)&0xF], 16), _mm512_srli_epi32(LFSR_S[ (9+buf)&0xF], 15));\
	BRC_X[2] = _mm512_or_si512(_mm512_slli_epi32(LFSR_S[ (7+buf)&0xF], 16), _mm512_srli_epi32(LFSR_S[ (5+buf)&0xF], 15));\
	BRC_X[3] = _mm512_or_si512(_mm512_slli_epi32(LFSR_S[ (2+buf)&0xF], 16), _mm512_srli_epi32(LFSR_S[ buf], 15))	

//#define V_SHIFT()\
//	LFSR_S[buf] = f;\
//	buf = (buf+1)&0xF

#define V_SHIFT()\
	LFSR_S[0] = LFSR_S[1];\
	LFSR_S[1] = LFSR_S[2];\
	LFSR_S[2] = LFSR_S[3];\
	LFSR_S[3] = LFSR_S[4];\
	LFSR_S[4] = LFSR_S[5];\
	LFSR_S[5] = LFSR_S[6];\
	LFSR_S[6] = LFSR_S[7];\
	LFSR_S[7] = LFSR_S[8];\
	LFSR_S[8] = LFSR_S[9];\
	LFSR_S[9] = LFSR_S[10];\
	LFSR_S[10] = LFSR_S[11];\
	LFSR_S[11] = LFSR_S[12];\
	LFSR_S[12] = LFSR_S[13];\
	LFSR_S[13] = LFSR_S[14];\
	LFSR_S[14] = LFSR_S[15];\
	LFSR_S[15] = f


#define V_FSM()\
	W = _mm512_add_epi32(_mm512_xor_si512(F_R[0], BRC_X[0]), F_R[1]);\
	W1 = _mm512_add_epi32(F_R[0], BRC_X[1]);\
	W2 = _mm512_xor_si512(F_R[1], BRC_X[2]);\
	u = _mm512_or_si512(_mm512_slli_epi32(W1, 16), _mm512_srli_epi32(W2, 16));\
	v = _mm512_or_si512(_mm512_slli_epi32(W2, 16), _mm512_srli_epi32(W1, 16));\
	u = _mm512_xor_si512(_mm512_xor_si512(_mm512_xor_si512(_mm512_xor_si512(u, V4ROTL(u, 2)), V4ROTL(u, 10)), V4ROTL(u, 18)), V4ROTL(u, 24));\
	v = _mm512_xor_si512(_mm512_xor_si512(_mm512_xor_si512(_mm512_xor_si512(v, V4ROTL(v, 8)), V4ROTL(v, 14)), V4ROTL(v, 22)), V4ROTL(v, 30));\
	a = sbox0(V4BLENDB(u, _mm512_srli_epi32(v, 8)));\
	b = sbox1(V4BLENDB(_mm512_slli_epi32(u, 8), v));\
	F_R[0] = V4BLENDB(a, _mm512_srli_epi32(b, 8));\
	F_R[1] = V4BLENDB(_mm512_slli_epi32(a, 8), b)

#define odd_byte_mask _mm512_set1_epi32(0x00FF00FF)
#define even_byte_mask _mm512_set1_epi32(0xFF00FF00)

#define MAKEU31(a, b, c, d) _mm512_or_si512(_mm512_or_si512(_mm512_or_si512(_mm512_slli_epi32(a, 23), _mm512_slli_epi32(b, 16)), _mm512_slli_epi32(c, 8)), d)
#define GATHER(a, b) _mm512_and_si512(_mm512_i32gather_epi32(a, b, 1), _mm512_set1_epi32(0xFF))

/* the constants d */
static const u8 EK_d[16] =
{
	0x22, 0x2F, 0x24, 0x2A, 0x6D, 0x40, 0x40, 0x40,
	0x40, 0x40, 0x40, 0x40, 0x40, 0x52, 0x10, 0x30
};

/* the constants MAC d */
const u8 EK_d_MAC[3 * 16] =
{
	0x22, 0x2F, 0x25, 0x2A, 0x6D, 0x40, 0x40, 0x40,
	0x40, 0x40, 0x40, 0x40, 0x40, 0x52, 0x10, 0x30,
	0x23, 0x2F, 0x24, 0x2A, 0x6D, 0x40, 0x40, 0x40,
	0x40, 0x40, 0x40, 0x40, 0x40, 0x52, 0x10, 0x30,
	0x23, 0x2F, 0x25, 0x2A, 0x6D, 0x40, 0x40, 0x40,
	0x40, 0x40, 0x40, 0x40, 0x40, 0x52, 0x10, 0x30
};

inline void ZUC256_LFSRINIT_AVX512(ZUC256_AVX512_State* state, const u8* k, const u8* iv, const u8* d)
{
	__m512i vindex0, vindex1;
	vindex0 = _mm512_setr_epi32(0, 32, 32 * 2, 32 * 3, 32 * 4, 32 * 5, 32 * 6, 32 * 7,
		32 * 8, 32 * 9, 32 * 10, 32 * 11, 32 * 12, 32 * 13, 32 * 14, 32 * 15);		//	15*32,14*32...32,0
	vindex1 = _mm512_setr_epi32(0, 25, 25 * 2, 25 * 3, 25 * 4, 25 * 5, 25 * 6, 25 * 7,
		25 * 8, 25 * 9, 25 * 10, 25 * 11, 25 * 12, 25 * 13, 25 * 14, 25 * 15);		//	15*25,14*25...25,0

	state->LFSR_S[0] = MAKEU31(GATHER(vindex0, k), _mm512_set1_epi32(d[0]), GATHER(vindex0, k + 21), GATHER(vindex0, k + 16));
	state->LFSR_S[1] = MAKEU31(GATHER(vindex0, k + 1), _mm512_set1_epi32(d[1]), GATHER(vindex0, k + 22), GATHER(vindex0, k + 17));
	state->LFSR_S[2] = MAKEU31(GATHER(vindex0, k + 2), _mm512_set1_epi32(d[2]), GATHER(vindex0, k + 23), GATHER(vindex0, k + 18));
	state->LFSR_S[3] = MAKEU31(GATHER(vindex0, k + 3), _mm512_set1_epi32(d[3]), GATHER(vindex0, k + 24), GATHER(vindex0, k + 19));
	state->LFSR_S[4] = MAKEU31(GATHER(vindex0, k + 4), _mm512_set1_epi32(d[4]), GATHER(vindex0, k + 25), GATHER(vindex0, k + 20));
	state->LFSR_S[5] = MAKEU31(GATHER(vindex1, iv), _mm512_or_si512(_mm512_set1_epi32(d[5]), GATHER(vindex1, iv + 17)),
		GATHER(vindex0, k + 5), GATHER(vindex0, k + 26));
	state->LFSR_S[6] = MAKEU31(GATHER(vindex1, iv + 1), _mm512_or_si512(_mm512_set1_epi32(d[6]), GATHER(vindex1, iv + 18)),
		GATHER(vindex0, k + 6), GATHER(vindex0, k + 27));
	state->LFSR_S[7] = MAKEU31(GATHER(vindex1, iv + 10), _mm512_or_si512(_mm512_set1_epi32(d[7]), GATHER(vindex1, iv + 19)),
		GATHER(vindex0, k + 7), GATHER(vindex1, iv + 2));
	state->LFSR_S[8] = MAKEU31(GATHER(vindex0, k + 8), _mm512_or_si512(_mm512_set1_epi32(d[8]), GATHER(vindex1, iv + 20)),
		GATHER(vindex1, iv + 3), GATHER(vindex1, iv + 11));
	state->LFSR_S[9] = MAKEU31(GATHER(vindex0, k + 9), _mm512_or_si512(_mm512_set1_epi32(d[9]), GATHER(vindex1, iv + 21)),
		GATHER(vindex1, iv + 12), GATHER(vindex1, iv + 4));
	state->LFSR_S[10] = MAKEU31(GATHER(vindex1, iv + 5), _mm512_or_si512(_mm512_set1_epi32(d[10]), GATHER(vindex1, iv + 22)),
		GATHER(vindex0, k + 10), GATHER(vindex0, k + 28));
	state->LFSR_S[11] = MAKEU31(GATHER(vindex0, k + 11), _mm512_or_si512(_mm512_set1_epi32(d[11]), GATHER(vindex1, iv + 23)),
		GATHER(vindex1, iv + 6), GATHER(vindex1, iv + 13));
	state->LFSR_S[12] = MAKEU31(GATHER(vindex0, k + 12), _mm512_or_si512(_mm512_set1_epi32(d[12]), GATHER(vindex1, iv + 24)),
		GATHER(vindex1, iv + 7), GATHER(vindex1, iv + 14));
	state->LFSR_S[13] = MAKEU31(GATHER(vindex0, k + 13), _mm512_set1_epi32(d[13]), GATHER(vindex1, iv + 15), GATHER(vindex1, iv + 8));
	state->LFSR_S[14] = MAKEU31(GATHER(vindex0, k + 14), _mm512_or_si512(_mm512_set1_epi32(d[14]), _mm512_srli_epi32(GATHER(vindex0, k + 31), 4)),
		GATHER(vindex1, iv + 16), GATHER(vindex1, iv + 9));
	state->LFSR_S[15] = MAKEU31(GATHER(vindex0, k + 15), _mm512_or_si512(_mm512_set1_epi32(d[15]), _mm512_and_si512(GATHER(vindex0, k + 31), _mm512_set1_epi32(0xF))),
		GATHER(vindex0, k + 30), GATHER(vindex0, k + 29));

	state->F_R[0] = _mm512_setzero_si512();
	state->F_R[1] = _mm512_setzero_si512();
}

void ZUC256_AVX512(u32* ks, int wordlen, const u8* k, const u8* iv)
{
	__m512i W, W1, W2, u, v, a, b, f;
	ZUC256_AVX512_State state;
	__m512i* LFSR_S = state.LFSR_S, * F_R = state.F_R, * BRC_X = state.BRC_X;
	int i;
	int buf = 0;
	ZUC256_Setup_AVX512();
	ZUC256_LFSRINIT_AVX512(&state, k, iv, EK_d);




	for (i = 0; i < 32; i++)
	{
		V_BITR();
		V_FSM();
		V_FEEDBACK(f, v);
		f = _mm512_add_epi32(f, _mm512_srli_epi32(W, 1));
		f = AMR(f);
		V_SHIFT();
	}

	V_BITR();
	V_FSM();

	for (i = 0; i < wordlen; i++)
	{

		V_FEEDBACK(f, v);
		V_SHIFT();

		V_BITR();
		V_FSM();
		v = _mm512_xor_si512(W, BRC_X[3]);		//	v是最后的密钥
		_mm512_storeu_si512(ks + 16 * i, v);
	}

}

void ZUC256_CRYPT_AVX512(u32* C, const u8* CK, const u8* IV, const u32* M, int LENGTH)
{
	__m512i W, W1, W2, u, v, a, b, f;
	__m512i sqt = _mm512_setzero_si512();
	ZUC256_AVX512_State state;
	__m512i* LFSR_S = state.LFSR_S, * F_R = state.F_R, * BRC_X = state.BRC_X;
	int i, buf = 0;

	ZUC256_Setup_AVX512();
	ZUC256_LFSRINIT_AVX512(&state, CK, IV, EK_d);

	for (i = 0; i < 32; i++)
	{
		V_BITR_INIT();
		V_FSM();
		V_FEEDBACK(f, v);
		f = _mm512_add_epi32(f, _mm512_srli_epi32(W, 1));
		f = AMR(f);
		sqt = f;
		V_SHIFT();
	}

	V_BITR_INIT();
	V_FSM();

	for (i = 0; i < LENGTH; i++)
	{
		V_FEEDBACK(f, v);
		V_SHIFT();
		V_BITR();
		V_FSM();
		v = _mm512_loadu_si512(M + 16 * i);
		v = _mm512_xor_si512(_mm512_xor_si512(W, BRC_X[3]), v);
		_mm512_storeu_si512(C + 16 * i, v);
	}
}


__m512i Word_Reverse(__m512i r)
{
	__m512i t;

	t = _mm512_xor_si512(_mm512_and_si512(_mm512_slli_epi32(r, 1), left_1bit_mask), _mm512_and_si512(_mm512_srli_epi32(r, 1), right_1bit_mask));
	t = _mm512_xor_si512(_mm512_and_si512(_mm512_slli_epi32(t, 2), left_2bits_mask), _mm512_and_si512(_mm512_srli_epi32(t, 2), right_2bits_mask));
	t = _mm512_xor_si512(_mm512_and_si512(_mm512_slli_epi32(t, 4), left_4bits_mask), _mm512_and_si512(_mm512_srli_epi32(t, 4), right_4bits_mask));
	t = _mm512_xor_si512(_mm512_and_si512(_mm512_slli_epi32(t, 8), left_8bits_mask), _mm512_and_si512(_mm512_srli_epi32(t, 8), right_8bits_mask));

	return _mm512_xor_si512(_mm512_slli_epi32(t, 16), _mm512_srli_epi32(t, 16));
}

void ZUC256_MAC_AVX512(u32* MAC, int MAC_BITLEN, const u8* IK, const u8* IV, const u32* M, const u32 LENGTH)
{
	__m512i W, W1, W2, u, v, a, b, f;
	//__m128i s[8], r[8], t;
	__m256i z[2];
	__m512i* vecz, t0, t1;
	__m512i tmp[5], temp[4] = { 0 };
	ZUC256_AVX512_State state;
	__m512i* LFSR_S = state.LFSR_S, * F_R = state.F_R, * BRC_X = state.BRC_X;
	u32 d_index = (MAC_BITLEN >> 6) << 4, MAC_WORDLEN = MAC_BITLEN >> 5, L = LENGTH + 2 * MAC_WORDLEN;
	u32 i, j, k;
	int buf = 0;
	__m512i sqt = _mm512_setzero_si512();

	vecz = (__m512i*)malloc(L * sizeof(__m512i));

	ZUC256_Setup_AVX512();
	ZUC256_LFSRINIT_AVX512(&state, IK, IV, EK_d_MAC + d_index);

	for (i = 0; i < 32; i++)
	{
		V_BITR();
		V_FSM();
		V_FEEDBACK(f, v);
		f = _mm512_add_epi32(f, _mm512_srli_epi32(W, 1));
		f = AMR(f);
		V_SHIFT();
	}

	V_BITR();
	V_FSM();

	for (i = 0; i < L; i++)
	{
		V_FEEDBACK(f, v);
		V_SHIFT();
		V_BITR();
		V_FSM();
		vecz[i] = _mm512_xor_si512(W, BRC_X[3]);
	}

	__m512i r[2], s[2], t, t2;
	__m512i idx = _mm512_set_epi32(
		15, 13, 11, 9, 7, 5, 3, 1,
		15, 13, 11, 9, 7, 5, 3, 1
	);
	__m512i combined, permuted;
	__m256i result[2];

	for (i = 0; i < LENGTH; i++)
	{
		t0 = Word_Reverse(_mm512_loadu_si512(M + 16 * i));

		z[0] = _mm512_castsi512_si256(t0);
		z[1] = _mm512_extracti32x8_epi32(t0, 1);
		t0 = _mm512_cvtepu32_epi64(z[0]);
		t1 = _mm512_cvtepu32_epi64(z[1]);

		r[0] = t0;
		r[1] = t1;

		for (j = 0; j < MAC_WORDLEN; j++)
		{
			t0 = _mm512_unpacklo_epi32(vecz[i + j + MAC_WORDLEN + 1], vecz[i + j + MAC_WORDLEN]);
			t1 = _mm512_unpackhi_epi32(vecz[i + j + MAC_WORDLEN + 1], vecz[i + j + MAC_WORDLEN]);

			s[0] = t0;
			s[1] = t1;

			for (k = 0; k < 2; k++)
			{
				t = _mm512_clmulepi64_epi128(s[k], r[k], 0x00);
				t2 = _mm512_clmulepi64_epi128(s[k], r[k], 0x11);

				combined = _mm512_unpacklo_epi64(t, t2);

				permuted = _mm512_permutexvar_epi32(idx, combined);
				result[k] = _mm512_extracti32x8_epi32(permuted, 0);
			}

			tmp[j] = _mm512_castsi256_si512(result[0]);
			tmp[j] = _mm512_inserti64x4(tmp[j], result[1], 1);

			tmp[j] = _mm512_xor_si512(tmp[j], temp[j]);
			temp[j] = tmp[j];
		}
	}

	for (i = 0; i < MAC_WORDLEN; i++)
	{
		tmp[i] = _mm512_xor_si512(_mm512_xor_si512(vecz[i], tmp[i]), vecz[LENGTH + MAC_WORDLEN + i]);
		_mm512_storeu_si512(MAC + 16 * i, tmp[i]);
	}
	free(vecz);
}
