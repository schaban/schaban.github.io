/* SPDX-License-Identifier: MIT */
/* SPDX-FileCopyrightText: 2020-2022 Sergey Chaban <sergey.chaban@gmail.com> */

#include "vema.h"

#ifdef VEMA_NO_CLIB
static void vemaMemCpy(void* pDst, const void* pSrc, const size_t len) {
	if (pDst && pSrc) {
		uint8_t* pd = (uint8_t*)pDst;
		const uint8_t* ps = (const uint8_t*)pSrc;
		size_t i;
		for (i = 0; i < len; ++i) {
			*pd++ = *ps++;
		}
	}
}
static void vemaMemSet(void* pDst, const int val, const size_t len) {
	if (pDst) {
		uint8_t* pd = (uint8_t*)pDst;
		uint8_t v = (uint8_t)val;
		size_t i;
		for (i = 0; i < len; ++i) {
			*pd++ = v;
		}
	}
}
#else
#include <math.h>
#include <string.h>
#define vemaMemCpy memcpy
#define vemaMemSet memset
#endif


#if defined(VEMA_GCC_SSE) || defined(VEMA_SSE) || defined(VEMA_AVX)
#	include <immintrin.h>
#	define VEMA_SIMD_MIX_MASK(_ix0, _iy0, _iz1, _iw1) ( (_ix0)|((_iy0)<<2)|((_iz1)<<4)|((_iw1)<<6) )
#	define VEMA_SIMD_ELEM_MASK(_idx) VEMA_SIMD_MIX_MASK(_idx, _idx, _idx, _idx)
#endif

#if defined (VEMA_NEON) || defined(VEMA_ARM_VSQRT)
#	include <arm_neon.h>
static float32x4_t vemaNeonElem(const float32x4_t v, const int idx) {
	int i;
	float32x4_t res;
	float x = v[idx];
	for (i = 0; i < 4; ++i) {
		res[i] = x;
	}
	return res;
}
#endif


VEMA_IFC(uint32_t, Float32Bits)(const float x) {
	uint32_t bits;
	vemaMemCpy(&bits, &x, sizeof(uint32_t));
	return bits;
}

VEMA_IFC(float, Float32FromBits)(const uint32_t bits) {
	float x;
	vemaMemCpy(&x, &bits, sizeof(float));
	return x;
}

VEMA_IFC(VemaHalf, FloatToHalf)(const float x) {
	uint32_t a;
	uint32_t b = VEMA_FN(Float32Bits)(x);
	uint16_t s = (uint16_t)((b >> 16) & (1 << 15));
	b &= ~(1U << 31);
	if (b > 0x477FE000U) {
		/* infinity */
		b = 0x7C00;
	} else {
		if (b < 0x38800000U) {
			uint32_t r = 0x70 + 1 - (b >> 23);
			b &= (1U << 23) - 1;
			b |= (1U << 23);
			b >>= r;
		} else {
			b += 0xC8000000U;
		}
		a = (b >> 13) & 1;
		b += (1U << 12) - 1;
		b += a;
		b >>= 13;
		b &= (1U << 15) - 1;
	}
	return (VemaHalf)(b | s);
}

VEMA_IFC(float, HalfToFloat)(const VemaHalf h) {
	uint32_t v = (h & ((1 << 16) - 1)) ? 0xFFFFFFFF : 0;
	uint32_t e = ((((h >> 10) & 0x1F) + 0x70) & v) << 23;
	uint32_t m = (h & ((1 << 10) - 1)) << (23 - 10);
	uint32_t s = (uint32_t)(h >> 15) << 31;
	return VEMA_FN(Float32FromBits)(e | m | s);
}

VEMA_IFC(void, HalfToFloatAry)(float* pDstF, const VemaHalf* pSrcH, const size_t n) {
	size_t i;
	if (!pDstF) return;
	if (!pSrcH) return;
	for (i = 0; i < n; ++i) {
		pDstF[i] = VEMA_FN(HalfToFloat)(pSrcH[i]);
	}
}

VEMA_IFC(void, FloatToU32Ary)(uint32_t* pDstU, const float* pSrcF, const size_t n) {
	size_t i;
	if (!pDstU) return;
	if (!pSrcF) return;
	for (i = 0; i < n; ++i) {
		pDstU[i] = (uint32_t)pSrcF[i];
	}
}

VEMA_IFC(void, FloatToU32AryOffs)(uint32_t* pDstU, const size_t dstOffs, const float* pSrcF, const size_t srcOffs, const size_t n) {
	size_t i;
	float f;
	uint32_t u;
	uint8_t* pDst = (uint8_t*)pDstU;
	uint8_t* pSrc = (uint8_t*)pSrcF;
	if (!pDst) return;
	if (!pSrc) return;
	for (i = 0; i < n; ++i) {
		vemaMemCpy(&f, pSrc, sizeof(float));
		u = (uint32_t)f;
		vemaMemCpy(pDst, &u, sizeof(uint32_t));
		pSrc += sizeof(float);
		pDst += sizeof(uint32_t);
	}
}

VEMA_IFC(void, FloatToU16Ary)(uint16_t* pDstU, const float* pSrcF, const size_t n) {
	size_t i;
	if (!pDstU) return;
	if (!pSrcF) return;
	for (i = 0; i < n; ++i) {
		pDstU[i] = (uint16_t)pSrcF[i];
	}
}

VEMA_IFC(void, FloatToU16AryOffs)(uint16_t* pDstU, const size_t dstOffs, const float* pSrcF, const size_t srcOffs, const size_t n) {
	size_t i;
	float f;
	uint16_t u;
	uint8_t* pDst = (uint8_t*)pDstU;
	uint8_t* pSrc = (uint8_t*)pSrcF;
	if (!pDst) return;
	if (!pSrc) return;
	for (i = 0; i < n; ++i) {
		vemaMemCpy(&f, pSrc, sizeof(float));
		u = (uint16_t)f;
		vemaMemCpy(pDst, &u, sizeof(uint16_t));
		pSrc += sizeof(float);
		pDst += sizeof(uint16_t);
	}
}



VEMA_IFC(int, AlmostEqualF)(const float x, const float y, const float tol) {
	int eq = 0;
	float adif = VEMA_FN(AbsF)(x - y);
	if (x == 0.0f || y == 0.0f) {
		eq = adif <= tol;
	} else {
		float ax = VEMA_FN(AbsF)(x);
		float ay = VEMA_FN(AbsF)(y);
		eq = (adif / ax) <= tol && (adif / ay) <= tol;
	}
	return eq;
}


VEMA_IFC(float, EvalPolynomial)(const float x, const float* pRevCoefs, const size_t n) {
	const float* p = pRevCoefs;
	float res = *p++;
	size_t i;
	for (i = 0; i < n; ++i) {
		res *= x;
		res += p[i];
	}
	return res;
}


VEMA_IFC(float, FrExpF)(const float x, int* pExp) {
	uint32_t bits = VEMA_FN(Float32Bits)(x);
	int e = (bits >> 23) & 0xFF;
	float fr = x;
	if (e == 0) {
		if (x != 0.0f) {
			bits = VEMA_FN(Float32Bits)(x * VEMA_FN(Float32FromBits)(0x5F800000));
			e = (bits >> 23) & 0xFF;
			if (e != 0 && e != 0xFF) {
				bits &= ~(0xFF << 23);
				bits |= 0x7E << 23;
				fr = VEMA_FN(Float32FromBits)(bits);
				e -= (0x7E + 64);
			}
		}
	} else if (e != 0xFF) {
		bits &= ~(0xFF << 23);
		bits |= 0x7E << 23;
		fr = VEMA_FN(Float32FromBits)(bits);
		e -= 0x7E;
	}
	if (pExp) {
		*pExp = e;
	}
	return fr;
}

VEMA_IFC(float, LdExpF)(const float x, const int e) {
	int ex = e;
	return x * VEMA_FN(Float32FromBits)((uint32_t)(ex + 0x7F) << 23);
}


VEMA_IFC(float, InvalidF)(void) {
	return VEMA_FN(Float32FromBits)(0xFFFFFFFF);
}

VEMA_IFC(float, RadiansF)(const float deg) {
	return deg * (VEMA_PI_F / 180.0f);
}

VEMA_IFC(float, DegreesF)(const float rad) {
	return rad * (180.0f / VEMA_PI_F);
}

VEMA_IFC(float, SqF)(const float x) {
	return x * x;
}

VEMA_IFC(float, CbF)(const float x) {
	return x * x * x;
}

VEMA_IFC(float, Div0F)(const float x, const float y) {
	return y != 0.0f ? x / y : 0.0f;
}

VEMA_IFC(float, Rcp0F)(const float x) {
	return VEMA_FN(Div0F)(1.0f, x);
}

VEMA_IFC(float, ModF)(const float x, const float y) {
#ifdef VEMA_NO_CLIB
	float res = 0.0f;
	if (y != 0.0f) {
		float d = (float)(int)(x / y);
		res = x - (d * y);
	}
	return res;
#else
#ifdef VEMA_NO_MATH_F
	return (float)fmod(x, y);
#else
	return fmodf(x, y);
#endif
#endif
}

VEMA_IFC(float, AbsF)(const float x) {
#ifdef VEMA_NO_CLIB
	return x < 0.0f ? -x : x;
#else
#ifdef VEMA_NO_MATH_F
	return (float)fabs(x);
#else
	return fabsf(x);
#endif
#endif
}

VEMA_IFC(float, MinF)(const float x, const float y) {
	return x < y ? x : y;
}

VEMA_IFC(float, MaxF)(const float x, const float y) {
	return x > y ? x : y;
}

VEMA_IFC(float, Min3F)(const float x, const float y, const float z) {
	return VEMA_FN(MinF)(VEMA_FN(MinF)(x, y), z);
}

VEMA_IFC(float, Max3F)(const float x, const float y, const float z) {
	return VEMA_FN(MaxF)(VEMA_FN(MaxF)(x, y), z);
}

VEMA_IFC(float, Min4F)(const float x, const float y, const float z, const float w) {
	return VEMA_FN(MinF)(VEMA_FN(MinF)(x, y), VEMA_FN(MinF)(z, w));
}

VEMA_IFC(float, Max4F)(const float x, const float y, const float z, const float w) {
	return VEMA_FN(MaxF)(VEMA_FN(MaxF)(x, y), VEMA_FN(MaxF)(z, w));
}

VEMA_IFC(float, FloorF)(const float x) {
#ifdef VEMA_NO_CLIB
	float res = (float)(int)x;
	if (x < 0.0f) {
		float frc = x - res;
		if (frc != 0.0f) {
			res -= 1.0f;
		}
	}
	return res;
#else
#ifdef VEMA_NO_MATH_F
	return (float)floor(x);
#else
	return floorf(x);
#endif
#endif
}

VEMA_IFC(float, CeilF)(const float x) {
#ifdef VEMA_NO_CLIB
	float res = (float)(int)x;
	if (x > 0.0f) {
		float frc = x - res;
		if (frc != 0.0f) {
			res += 1.0f;
		}
	}
	return res;
#else
#ifdef VEMA_NO_MATH_F
	return (float)ceil(x);
#else
	return ceilf(x);
#endif
#endif
}

VEMA_IFC(float, RoundF)(const float x) {
	float val = x;
	int sgn = x < 0.0f;
	float ival;
	float frc;
	float res;
	if (sgn) {
		val = -val;
	}
	ival = (float)((int32_t)val);
	frc = val - ival;
	res = ival;
	if (frc >= 0.5f) {
		res += 1.0f;
	}
	if (sgn) {
		res = -res;
	}
	return res;
}

VEMA_IFC(float, TruncF)(const float x) {
	return VEMA_FN(FloorF)(x);
}

VEMA_IFC(float, SqrtF)(const float x) {
#if defined(VEMA_GCC_SSE)
	return __builtin_ia32_sqrtss(_mm_set_ss(x))[0];
#elif defined(VEMA_SSE) || defined(VEMA_AVX)
	float res;
	_mm_store_ss(&res, _mm_sqrt_ss(_mm_set_ss(x)));
	return res;
#elif defined(VEMA_GCC_AARCH64_ASM)
	float res;
	__asm__("fsqrt %s0, %s1" : "=w"(res) : "w"(x));
	return res;
#elif defined(VEMA_ARM_VSQRT)
	float32x2_t xx = { x, x };
	return vsqrt_f32(xx)[0];
#elif defined(VEMA_GCC_BUILTINS)
	return __builtin_sqrtf(x);
#elif defined(VEMA_NO_CLIB)
	float y = 0.0f;
	if (x > 0.0f) {
		int ex;
		float fr = VEMA_FN(FrExpF)(x, &ex);
		if (ex & 1) {
			fr *= 2.0f;
			--ex;
		}
		ex >>= 1;
		if (fr > 1.41421356237f) {
			y = fr - 2.0f;
			y = (((((-0.000988430693f*y + 0.000794799533f)*y - 0.00358905364f)*y + 0.0110288095f)*y - 0.044195205f)*y + 0.353553385f)*y + 1.41421354f;
		} else if (fr > 0.707106781187f) {
			y = fr - 1.0f;
			y = (((((0.0135199288f*y - 0.0226657763f)*y + 0.0278720781f)*y - 0.0389582776f)*y + 0.0624811128f)*y - 0.125001505f)*y*y + 0.5f*y + 1.0f;
		} else {
			y = fr - 0.5f;
			y = (((((-0.394950062f*y + 0.517430365f)*y - 0.432144374f)*y + 0.353107303f)*y - 0.353545815f)*y + 0.707106769f)*y + 0.707106829f;
		}
		y = VEMA_FN(LdExpF)(y, ex);
	}
	return y;
#else
#ifdef VEMA_NO_MATH_F
	return (float)sqrt(x);
#else
	return sqrtf(x);
#endif
#endif
}

VEMA_IFC(float, SinF)(const float x) {
#ifdef VEMA_NO_CLIB
	float val;
	float sgn;
	uint32_t ir;
	float fr;
	float s;
	float y;
	if (x < 0.0f) {
		sgn = -1.0f;
		val = -x;
	} else {
		sgn = 1.0f;
		val = x;
	}
	if (val > (float)((1<<24)-1)) {
		return 0.0f;
	}
	ir = (uint32_t)(val * (4.0f / VEMA_PI_F));
	ir += ir & 1;
	fr = (float)ir;
	ir &= 7;
	if (ir > 3) {
		ir -= 4;
		sgn = -sgn;
	}
	if (val > (float)(1 << 13)) {
		val = val - (fr * (VEMA_PI_F / 4.0f));
	} else {
		val -= fr * 0.78515625f;
		val -= fr * 0.000241875648f;
		val -= fr * 3.77489506e-8f;
	}
	s = val * val;
	switch (ir) {
		case 1:
		case 2:
			y = ((s*2.44331568e-5f - 0.00138873165f)*s + 0.0416666456f) * s*s;
			y -= s * 0.5f;
			y += 1.0f;
			break;
		default:
			y = ((0.00833216123f - s*0.000195152956f)*s - 0.166666552f)*s*val + val;
			break;
	}
	y *= sgn;
	return y;
#else
#ifdef VEMA_NO_TRIG_F
	return (float)sin(x);
#else
	return sinf(x);
#endif
#endif
}

VEMA_IFC(float, CosF)(const float x) {
#ifdef VEMA_NO_CLIB
	float val;
	float sgn;
	uint32_t ir;
	float fr;
	float s;
	float y;
	if (x < 0.0f) {
		val = -x;
	} else {
		val = x;
	}
	if (val >(float)((1 << 24) - 1)) {
		return 0.0f;
	}
	sgn = 1.0f;
	ir = (uint32_t)(val * (4.0f / VEMA_PI_F));
	ir += ir & 1;
	fr = (float)ir;
	ir &= 7;
	if (ir > 3) {
		ir -= 4;
		sgn = -sgn;
	}
	if (ir > 1) {
		sgn = -sgn;
	}
	if (val > (float)(1 << 13)) {
		val = val - (fr * (VEMA_PI_F / 4.0f));
	} else {
		val -= fr * 0.78515625f;
		val -= fr * 0.000241875648f;
		val -= fr * 3.77489506e-8f;
	}
	s = val * val;
	switch (ir) {
		case 1:
		case 2:
			y = ((0.00833216123f - s*0.000195152956f)*s - 0.166666552f)*s*val + val;
			break;
		default:
			y = ((s*2.44331568e-5f - 0.00138873165f)*s + 0.0416666456f) * s*s;
			y -= s * 0.5f;
			y += 1.0f;
			break;
	}
	y *= sgn;
	return y;
#else
#ifdef VEMA_NO_TRIG_F
	return (float)cos(x);
#else
	return cosf(x);
#endif
#endif
}

VEMA_IFC(float, TanF)(const float x) {
#ifdef VEMA_NO_CLIB
	float val;
	float sgn = 0.0f;
	float y = 0.0f;
	if (x < 0.0f) {
		sgn = -1.0f;
		val = -x;
	} else {
		sgn = 1.0f;
		val = x;
	}
	if (val <= (float)(1 << 13)) {
		uint32_t ir;
		float fr;
		ir = (uint32_t)(val * (4.0f / VEMA_PI_F));
		ir += ir & 1;
		fr = (float)ir;
		y = val;
		val -= fr * 0.78515625f;
		val -= fr * 0.000241875648f;
		val -= fr * 3.77489506e-8f;
		if (y > 1.0e-4f) {
			float s = val * val;
			y = (((((s*0.00938540231f + 0.00311992224f)*s + 0.0244301353f)*s + 0.0534112789f)*s + 0.133387998f)*s + 0.333331555f)*s*y + y;
		} else {
			y = val;
		}
		if (ir & 2) {
			if (y != 0.0f) {
				y = -1.0f / y;
			}
		}
	}
	y *= sgn;
	return y;
#else
#ifdef VEMA_NO_TRIG_F
	return (float)tan(x);
#else
	return tanf(x);
#endif
#endif
}

VEMA_IFC(float, ArcSinF)(const float x) {
#ifdef VEMA_NO_CLIB
	float val;
	float sgn = 0.0f;
	float y = 0.0f;
	int flg = 0;
	if (x < 0.0f) {
		sgn = -1.0f;
		val = -x;
	} else {
		sgn = 1.0f;
		val = x;
	}
	if (val < 1.0f) {
		float t;
		y = val;
		if (val >= 1.0e-4f) {
			if (val > 0.5f) {
				y = (1.0f - val) * 0.5f;
				t = VEMA_FN(SqrtF)(y);
				flg = 1;
			} else {
				y = val * val;
				t = val;
			}
			y = ((((y*0.0421632007f + 0.024181312f)*y + 0.0454700254f)*y + 0.0749530047f)*y + 0.166667521f)*y*t + t;
			if (flg) {
				y = (VEMA_PI_F / 2.0f) - y*2.0f;
			}
		}
	}
	y *= sgn;
	return y;
#else
#ifdef VEMA_NO_TRIG_F
	return (float)asin(x);
#else
	return asinf(x);
#endif
#endif
}

VEMA_IFC(float, ArcCosF)(const float x) {
#ifdef VEMA_NO_CLIB
	float y = 0.0f;
	if (x >= -1.0f && x <= 1.0f) {
		if (x < -0.5f) {
			y = VEMA_PI_F - 2.0f*VEMA_FN(ArcSinF)(VEMA_FN(SqrtF)((x + 1.0f)*0.5f));
		} else if (x > 0.5f) {
			y = 2.0f*VEMA_FN(ArcSinF)(VEMA_FN(SqrtF)((1.0f - x)*0.5f));
		} else {
			y = VEMA_PI_F*0.5f - VEMA_FN(ArcSinF)(x);
		}
	}
	return y;
#else
#ifdef VEMA_NO_TRIG_F
	return (float)acos(x);
#else
	return acosf(x);
#endif
#endif
}

VEMA_IFC(float, ArcTanF)(const float x) {
#ifdef VEMA_NO_CLIB
	float val;
	float sgn = 0.0f;
	float y = 0.0f;
	float s;
	if (x < 0.0f) {
		sgn = -1.0f;
		val = -x;
	} else {
		sgn = 1.0f;
		val = x;
	}
	if (val > 2.41421366f) {
		y = VEMA_PI_F / 2.0f;
		val = -1.0f / val;
	} else if (val > 0.414213568f) {
		y = VEMA_PI_F / 4.0f;
		val = (val - 1.0f) / (val + 1.0f);
	}
	s = val * val;
	y += (((s*0.0805374458f - 0.138776854f)*s + 0.199777111f)*s - 0.333329499f)*s*val + val;
	y *= sgn;
	return y;
#else
#ifdef VEMA_NO_TRIG_F
	return (float)atan(x);
#else
	return atanf(x);
#endif
#endif
}

VEMA_IFC(float, ArcTan2F)(const float y, const float x) {
#ifdef VEMA_NO_CLIB
	float res = 0.0f;
	int sgn = (x < 0.0f ? 2 : 0) | (y < 0.0f ? 1 : 0);
	if (x == 0.0f) {
		if (sgn & 1) {
			res = -VEMA_PI_F / 2.0f;
		} else {
			if (y != 0.0f) {
				res = VEMA_PI_F / 2.0f;
			}
		}
	} else if (y == 0.0f) {
		if (sgn & 2) {
			res = VEMA_PI_F;
		}
	} else {
		switch (sgn) {
			case 2:
				res = VEMA_PI_F;
				break;
			case 3:
				res = -VEMA_PI_F;
				break;
			default:
				break;
		}
		res += VEMA_FN(ArcTanF)(y / x);
	}
	return res;
#else
#ifdef VEMA_NO_TRIG_F
	return (float)atan2(y, x);
#else
	return atan2f(y, x);
#endif
#endif
}

VEMA_IFC(float, SincF)(const float x) {
	if (VEMA_FN(AbsF)(x) < 1.0e-4f) {
		return 1.0f;
	}
	return VEMA_FN(SinF)(x) / x;
}

VEMA_IFC(float, InvSincF)(const float x) {
	if (VEMA_FN(AbsF)(x) < 1.0e-4f) {
		return 1.0f;
	}
	return x / VEMA_FN(SinF)(x);
}

VEMA_IFC(float, HypotF)(const float x, const float y) {
	float m = VEMA_FN(MaxF)(VEMA_FN(AbsF)(x), VEMA_FN(AbsF)(y));
	float im = VEMA_FN(Rcp0F)(m);
	return VEMA_FN(SqrtF)(VEMA_FN(SqF)(x*im) + VEMA_FN(SqF)(y*im)) * m;
}

VEMA_IFC(float, Log10F)(const float x) {
	return VEMA_FN(InvalidF)();
}

VEMA_IFC(float, LogF)(const float x) {
#ifdef VEMA_NO_CLIB
	float res = 0.0f;
	if (x > 0.0f) {
		int iex = 0;
		float t = VEMA_FN(FrExpF)(x, &iex);
		float tt;
		if (t < 0.707106769f) {
			t += t;
			--iex;
		}
		t -= 1.0f;
		tt = t * t;
		res = ((((((((t*0.0703768358f - 0.115146101f)*t + 0.116769984f)*t - 0.124201410f)*t + 0.142493233f)*t - 0.166680574f)*t + 0.200007141f)*t - 0.24999994f)*t + 0.333333313f) * t * tt;
		if (iex) {
			res += -0.000212194442f * (float)iex;
		}
		res += -0.5f * tt;
		res += t;
		if (iex) {
			res += 0.693359375f * (float)iex;
		}
	}
	return res;
#else
#ifdef VEMA_NO_MATH_F
	return (float)log(x);
#else
	return logf(x);
#endif
#endif
}

VEMA_IFC(float, ExpF)(const float x) {
#ifdef VEMA_NO_CLIB
	float e, t;
	if (x < -103.278931f || x > 88.7228394f) {
		return 0.0f;
	}
	e = VEMA_FN(FloorF)(x*1.44269502f + 0.5f);
	t = x;
	t -= e * 0.693359375f;
	t += e * 0.000212194442f;
	t = (((((t*0.000198756912f + 0.00139819994f)*t + 0.00833345205f)*t + 0.0416657962f)*t + 0.166666657f)*t + 0.5f)*t*t + t + 1.0f;
	return VEMA_FN(LdExpF)(t, (int)e);
#else
#ifdef VEMA_NO_MATH_F
	return (float)exp(x);
#else
	return expf(x);
#endif
#endif
}

VEMA_IFC(float, PowF)(const float x, const float y) {
#ifdef VEMA_NO_CLIB
	float res = 0.0f;
	int iex = (int)y;
	if ((float)iex == y) {
		res = VEMA_FN(IntPowF)(x, iex);
	} else {
		if (y == 0.5f) {
			res = VEMA_FN(SqrtF)(x);
		} else {
			res = VEMA_FN(ExpF)(VEMA_FN(LogF)(x) * y);
		}
	}
	return res;
#else
#ifdef VEMA_NO_MATH_F
	return (float)pow(x, y);
#else
	return powf(x, y);
#endif
#endif
}

VEMA_IFC(float, IntPowF)(const float x, const int n) {
	float res = 1.0f;
	float wx = x;
	int wn = n;
	if (n < 0) {
		wx = VEMA_FN(Rcp0F)(x);
		wn = -n;
	}
	do {
		if ((wn & 1) != 0) {
			res *= wx;
		}
		wx *= wx;
		wn >>= 1;
	} while (wn > 0);
	return res;
}

VEMA_IFC(float, LerpF)(const float a, const float b, const float t) {
	return a + (b - a)*t;
}

VEMA_IFC(float, ClampF)(const float x, const float lo, const float hi) {
	return VEMA_FN(MaxF)(VEMA_FN(MinF)(x, hi), lo);
}

VEMA_IFC(float, SaturateF)(const float x) {
	return VEMA_FN(ClampF)(x, 0.0f, 1.0f);
}

VEMA_IFC(float, FitF)(const float val, const float oldMin, const float oldMax, const float newMin, const float newMax) {
	float rel = VEMA_FN(Div0F)(val - oldMin, oldMax - oldMin);
	rel = VEMA_FN(SaturateF)(rel);
	return VEMA_FN(LerpF)(newMin, newMax, rel);
}

VEMA_IFC(float, EaseCurveF)(const float p1, const float p2, const float t) {
	float tt = t*t;
	float ttt = tt*t;
	float ttt3 = ttt*3.0f;
	float b1 = ttt3 - tt*6.0f + t*3.0f;
	float b2 = tt*3.0f - ttt3;
	return b1*p1 + b2*p2 + ttt;
}


VEMA_IFC(double, RadiansD)(const double deg) {
	return deg * (VEMA_PI_D / 180.0);
}

VEMA_IFC(double, DegreesD)(const double rad) {
	return rad * (180.0 / VEMA_PI_D);
}

VEMA_IFC(double, SqD)(const double x) {
	return x * x;
}

VEMA_IFC(double, CbD)(const double x) {
	return x * x * x;
}

VEMA_IFC(double, Div0D)(const double x, const double y) {
	return y != 0.0 ? x / y : 0.0;
}

VEMA_IFC(double, Rcp0D)(const double x) {
	return VEMA_FN(Div0D)(1.0, x);
}

VEMA_IFC(double, ModD)(const double x, const double y) {
#ifdef VEMA_NO_CLIB
	double res = 0.0f;
	if (y != 0.0f) {
		double d = (double)(int)(x / y);
		res = x - (d * y);
	}
	return res;
#else
	return fmod(x, y);
#endif
}

VEMA_IFC(double, AbsD)(const double x) {
#ifdef VEMA_NO_CLIB
	return x < 0.0 ? -x : x;
#else
	return fabs(x);
#endif
}

VEMA_IFC(double, MinD)(const double x, const double y) {
	return x < y ? x : y;
}

VEMA_IFC(double, MaxD)(const double x, const double y) {
	return x > y ? x : y;
}

VEMA_IFC(double, Min3D)(const double x, const double y, const double z) {
	return VEMA_FN(MinD)(VEMA_FN(MinD)(x, y), z);
}

VEMA_IFC(double, Max3D)(const double x, const double y, const double z) {
	return VEMA_FN(MaxD)(VEMA_FN(MaxD)(x, y), z);
}

VEMA_IFC(double, Min4D)(const double x, const double y, const double z, const double w) {
	return VEMA_FN(MinD)(VEMA_FN(MinD)(x, y), VEMA_FN(MinD)(z, w));
}

VEMA_IFC(double, Max4D)(const double x, const double y, const double z, const double w) {
	return VEMA_FN(MaxD)(VEMA_FN(MaxD)(x, y), VEMA_FN(MaxD)(z, w));
}

VEMA_IFC(double, FloorD)(const double x) {
#ifdef VEMA_NO_CLIB
	double res = (double)(int)x;
	if (x < 0.0f) {
		double frc = x - res;
		if (frc != 0.0f) {
			res -= 1.0f;
		}
	}
	return res;
#else
	return floor(x);
#endif
}

VEMA_IFC(double, CeilD)(const double x) {
#ifdef VEMA_NO_CLIB
	double res = (double)(int)x;
	if (x > 0.0f) {
		double frc = x - res;
		if (frc != 0.0f) {
			res += 1.0f;
		}
	}
	return res;
#else
	return ceil(x);
#endif
}

VEMA_IFC(double, RoundD)(const double x) {
	return VEMA_FN(FloorD)(x);
}

VEMA_IFC(double, SqrtD)(const double x) {
#if defined(VEMA_GCC_BUILTINS)
	return __builtin_sqrt(x);
#elif defined(VEMA_ARM_VSQRT)
	float64x1_t xx = { x };
	return vsqrt_f64(xx)[0];
#elif defined(VEMA_NO_CLIB)
	return VEMA_FN(SqrtF)((float)x);
#else
	return sqrt(x);
#endif
}

VEMA_IFC(double, PowD)(const double x, const double y) {
	return VEMA_FN(PowF)((float)x, (float)y);
}

VEMA_IFC(double, SinD)(const double x) {
	return VEMA_FN(SinF)((float)x);
}

VEMA_IFC(double, CosD)(const double x) {
	return VEMA_FN(CosF)((float)x);
}

VEMA_IFC(double, ArcSinD)(const double x) {
	return VEMA_FN(ArcSinF)((float)x);
}

VEMA_IFC(double, ArcCosD)(const double x) {
	return VEMA_FN(ArcCosF)((float)x);
}


#if defined(_MSC_VER) || (defined(__GNUC__) && !defined(__clang__))
#	define VEMA_PREC_VOLATILE
#else
#	define VEMA_PREC_VOLATILE volatile
#endif

#if defined(__GNUC__) && !defined(__clang__)
#	pragma GCC push_options
#	pragma GCC optimize("no-unsafe-math-optimizations")
#endif

VEMA_IFC(void, TwoSumF)(VemaVec2F ts, const float x, const float y) {
	VEMA_PREC_VOLATILE float s = x + y;
	VEMA_PREC_VOLATILE float xp = s - y;
	float yp = s - xp;
	VEMA_PREC_VOLATILE float dx = x - xp;
	float dy = y - yp;
	float t = dx + dy;
	ts[0] = s;
	ts[1] = t;
}

#if defined(__GNUC__) && !defined(__clang__)
#	pragma GCC pop_options
#endif


VEMA_IFC(void, SqrtVecF)(float* pDst, const int N) {
#ifdef VEMA_GCC_SSE
	int i;
	int n = N >> 2;
	for (i = 0; i < n; ++i) {
		*((__m128_u*)pDst) = __builtin_ia32_sqrtps(*((__m128_u*)pDst));
		pDst += 4;
	}
	n = N & 3;
	for (i = 0; i < n; ++i) {
		pDst[i] = VEMA_FN(SqrtF)(pDst[i]);
	}
#else
	int i;
	for (i = 0; i < N; ++i) {
		pDst[i] = VEMA_FN(SqrtF)(pDst[i]);
	}
#endif
}

VEMA_IFC(void, MulMtxF)(float* pDst, const float* pSrc1, const float* pSrc2, const int M, const int N, const int P) {
	int i, j, k;
	for (i = 0; i < M; ++i) {
		int ra = i * N;
		int rr = i * P;
		float s = pSrc1[ra];
		for (k = 0; k < P; ++k) {
			pDst[rr + k] = pSrc2[k] * s;
		}
	}
	for (i = 0; i < M; ++i) {
		int ra = i * N;
		int rr = i * P;
		for (j = 1; j < N; ++j) {
			int rb = j * P;
			float s = pSrc1[ra + j];
			for (k = 0; k < P; ++k) {
				pDst[rr + k] += pSrc2[rb + k] * s;
			}
		}
	}
}

VEMA_IFC(void, MulVecMtxF)(float* pDstVec, const float* pSrcVec, const float* pMtx, const int M, const int N) {
	VEMA_FN(MulMtxF)(pDstVec, pSrcVec, pMtx, 1, M, N);
}

VEMA_IFC(void, MulMtxVecF)(float* pDstVec, const float* pMtx, const float* pSrcVec, const int M, const int N) {
	VEMA_FN(MulMtxF)(pDstVec, pMtx, pSrcVec, M, N, 1);
}

VEMA_IFC(int, GJInvertMtxF)(float* pMtx, const int N, int* pWk /* [N*3] */) {
	int i, j, k;
	int* pPiv = pWk;
	int* pCol = pWk + N;
	int* pRow = pWk + N * 2;
	int ir = 0;
	int ic = 0;
	int rc;
	float piv;
	float ipiv;
	for (i = 0; i < N; ++i) {
		pPiv[i] = 0;
	}
	for (i = 0; i < N; ++i) {
		float amax = 0.0f;
		for (j = 0; j < N; ++j) {
			if (pPiv[j] != 1) {
				int rj = j * N;
				for (k = 0; k < N; ++k) {
					if (0 == pPiv[k]) {
						float a = pMtx[rj + k];
						if (a < 0.0f) a = -a;
						if (a >= amax) {
							amax = a;
							ir = j;
							ic = k;
						}
					}
				}
			}
		}
		++pPiv[ic];
		if (ir != ic) {
			int rr = ir * N;
			int rc = ic * N;
			for (j = 0; j < N; ++j) {
				float t = pMtx[rr + j];
				pMtx[rr + j] = pMtx[rc + j];
				pMtx[rc + j] = t;
			}
		}
		pRow[i] = ir;
		pCol[i] = ic;
		rc = ic * N;
		piv = pMtx[rc + ic];
		if (piv == 0) return 0; /* singular */
		ipiv = 1.0f / piv;
		pMtx[rc + ic] = 1;
		for (j = 0; j < N; ++j) {
			pMtx[rc + j] *= ipiv;
		}
		for (j = 0; j < N; ++j) {
			if (j != ic) {
				float* pDst;
				float* pSrc;
				int rj = j * N;
				float d = pMtx[rj + ic];
				pMtx[rj + ic] = 0;
				pDst = &pMtx[rj];
				pSrc = &pMtx[rc];
				for (k = 0; k < N; ++k) {
					*pDst++ -= *pSrc++ * d;
				}
			}
		}
	}
	for (i = N; --i >= 0;) {
		ir = pRow[i];
		ic = pCol[i];
		if (ir != ic) {
			for (j = 0; j < N; ++j) {
				int rj = j * N;
				float t = pMtx[rj + ir];
				pMtx[rj + ir] = pMtx[rj + ic];
				pMtx[rj + ic] = t;
			}
		}
	}
	return 1;
}

VEMA_IFC(int, LUDecompMtxF)(float* pMtx, const int N, float* pWk /* [N] */, int* pIdx /* [N] */, int* pDetSgn) {
	int i, j, k;
	int imax;
	int dsgn = 1;
	float* pScl = pWk;
	for (i = 0; i < N; ++i) {
		float scl = 0.0f;
		int ri = i * N;
		for (j = 0; j < N; ++j) {
			float a = pMtx[ri + j];
			if (a < 0) a = -a;
			if (a > scl) scl = a;
		}
		if (scl == 0) {
			if (pDetSgn) {
				*pDetSgn = 0;
			}
			return 0;
		}
		pScl[i] = scl;
	}
	for (i = 0; i < N; ++i) {
		pScl[i] = 1.0f / pScl[i];
	}
	imax = 0;
	for (k = 0; k < N; ++k) {
		int rk = k * N;
		float amax = 0.0f;
		for (i = k; i < N; ++i) {
			float a = pMtx[(i * N) + k];
			if (a < 0.0f) a = -a;
			a *= pScl[i];
			if (amax <= a) {
				amax = a;
				imax = i;
			}
		}
		if (k != imax) {
			int rm = imax * N;
			for (j = 0; j < N; ++j) {
				float t = pMtx[rm + j];
				pMtx[rm + j] = pMtx[rk + j];
				pMtx[rk + j] = t;
			}
			dsgn = -dsgn;
			pScl[imax] = pScl[k];
		}
		if (pIdx) {
			pIdx[k] = imax;
		}
		if (pMtx[rk + k] == 0.0f) {
			pMtx[rk + k] = 1.0e-15f;
		}
		for (i = k + 1; i < N; ++i) {
			float* pDst;
			float* pSrc;
			int ri = i * N;
			float s = pMtx[ri + k] / pMtx[rk + k];
			pMtx[ri + k] = s;
			pDst = &pMtx[ri + k + 1];
			pSrc = &pMtx[rk + k + 1];
			for (j = k + 1; j < N; ++j) {
				*pDst++ -= *pSrc++ * s;
			}
		}
	}
	if (pDetSgn) {
		*pDetSgn = dsgn;
	}
	return 1;
}


VEMA_IFC(void, MulMtxD)(double* pDst, const double* pSrc1, const double* pSrc2, const int M, const int N, const int P) {
	int i, j, k;
	for (i = 0; i < M; ++i) {
		int ra = i * N;
		int rr = i * P;
		double s = pSrc1[ra];
		for (k = 0; k < P; ++k) {
			pDst[rr + k] = pSrc2[k] * s;
		}
	}
	for (i = 0; i < M; ++i) {
		int ra = i * N;
		int rr = i * P;
		for (j = 1; j < N; ++j) {
			int rb = j * P;
			double s = pSrc1[ra + j];
			for (k = 0; k < P; ++k) {
				pDst[rr + k] += pSrc2[rb + k] * s;
			}
		}
	}
}

VEMA_IFC(void, MulVecMtxD)(double* pDstVec, const double* pSrcVec, const double* pMtx, const int M, const int N) {
	VEMA_FN(MulMtxD)(pDstVec, pSrcVec, pMtx, 1, M, N);
}

VEMA_IFC(void, MulMtxVecD)(double* pDstVec, const double* pMtx, const double* pSrcVec, const int M, const int N) {
	VEMA_FN(MulMtxD)(pDstVec, pMtx, pSrcVec, M, N, 1);
}


VEMA_IFC(void, ZeroVec2F)(VemaVec2F v) {
	vemaMemSet(v, 0, sizeof(VemaVec2F));
}

VEMA_IFC(void, CopyVec2F)(VemaVec2F dst, const VemaVec2F src) {
	if (dst != src) {
		vemaMemCpy(dst, src, sizeof(VemaVec2F));
	}
}

VEMA_IFC(void, LoadVec2F)(VemaVec2F v, const void* pMem) {
	if (pMem) {
		vemaMemCpy(v, pMem, sizeof(VemaVec2F));
	}
}

VEMA_IFC(void, StoreVec2F)(void* pMem, const VemaVec2F v) {
	if (pMem) {
		vemaMemCpy(pMem, v, sizeof(VemaVec2F));
	}
}

VEMA_IFC(void, LoadAtIdxVec2F)(VemaVec2F v, const void* pMem, const int32_t idx) {
	if (pMem) {
		VEMA_FN(LoadVec2F)(v, (const VemaVec2F*)pMem + idx);
	}
}

VEMA_IFC(void, StoreAtIdxVec2F)(void* pMem, const int32_t idx, const VemaVec2F v) {
	if (pMem) {
		VEMA_FN(StoreVec2F)((VemaVec2F*)pMem + idx, v);
	}
}

VEMA_IFC(void, LoadAtOffsVec2F)(VemaVec2F v, const void* pMem, const size_t offs) {
	if (pMem) {
		VEMA_FN(LoadVec2F)(v, (const VemaVec2F*)((const uint8_t*)pMem + offs));
	}
}

VEMA_IFC(void, StoreAtOffsVec2F)(void* pMem, const size_t offs, const VemaVec2F v) {
	if (pMem) {
		VEMA_FN(StoreVec2F)((VemaVec2F*)((uint8_t*)pMem + offs), v);
	}
}


VEMA_IFC(void, ZeroVec3F)(VemaVec3F v) {
	vemaMemSet(v, 0, sizeof(VemaVec3F));
}

VEMA_IFC(void, FillVec3F)(VemaVec3F v, const float s) {
	int i;
	for (i = 0; i < 3; ++i) {
		v[i] = s;
	}
}

VEMA_IFC(void, CopyVec3F)(VemaVec3F dst, const VemaVec3F src) {
	if (dst != src) {
		vemaMemCpy(dst, src, sizeof(VemaVec3F));
	}
}

VEMA_IFC(void, SetVec3F)(VemaVec3F v, const float x, const float y, const float z) {
	v[0] = x;
	v[1] = y;
	v[2] = z;
}

VEMA_IFC(void, LoadVec3F)(VemaVec3F v, const void* pMem) {
	if (pMem) {
		vemaMemCpy(v, pMem, sizeof(VemaVec3F));
	}
}

VEMA_IFC(void, StoreVec3F)(void* pMem, const VemaVec3F v) {
	if (pMem) {
		vemaMemCpy(pMem, v, sizeof(VemaVec3F));
	}
}

VEMA_IFC(void, LoadAtIdxVec3F)(VemaVec3F v, const void* pMem, const int32_t idx) {
	if (pMem) {
		VEMA_FN(LoadVec3F)(v, (const VemaVec3F*)pMem + idx);
	}
}

VEMA_IFC(void, StoreAtIdxVec3F)(void* pMem, const int32_t idx, const VemaVec3F v) {
	if (pMem) {
		VEMA_FN(StoreVec3F)((VemaVec3F*)pMem + idx, v);
	}
}

VEMA_IFC(void, LoadAtOffsVec3F)(VemaVec3F v, const void* pMem, const size_t offs) {
	if (pMem) {
		VEMA_FN(LoadVec3F)(v, (const VemaVec3F*)((const uint8_t*)pMem + offs));
	}
}

VEMA_IFC(void, StoreAtOffsVec3F)(void* pMem, const size_t offs, const VemaVec3F v) {
	if (pMem) {
		VEMA_FN(StoreVec3F)((VemaVec3F*)((uint8_t*)pMem + offs), v);
	}
}

VEMA_IFC(void, AddVec3F)(VemaVec3F v0, const VemaVec3F v1, const VemaVec3F v2) {
	int i;
	for (i = 0; i < 3; ++i) {
		v0[i] = v1[i] + v2[i];
	}
}

VEMA_IFC(void, SubVec3F)(VemaVec3F v0, const VemaVec3F v1, const VemaVec3F v2) {
	int i;
	for (i = 0; i < 3; ++i) {
		v0[i] = v1[i] - v2[i];
	}
}

VEMA_IFC(void, MulVec3F)(VemaVec3F v0, const VemaVec3F v1, const VemaVec3F v2) {
	int i;
	for (i = 0; i < 3; ++i) {
		v0[i] = v1[i] * v2[i];
	}
}

VEMA_IFC(void, CrossVec3F)(VemaVec3F v0, const VemaVec3F v1, const VemaVec3F v2) {
	float v10 = v1[0];
	float v11 = v1[1];
	float v12 = v1[2];
	float v20 = v2[0];
	float v21 = v2[1];
	float v22 = v2[2];
	v0[0] = v11*v22 - v12*v21;
	v0[1] = v12*v20 - v10*v22;
	v0[2] = v10*v21 - v11*v20;
}

VEMA_IFC(void, ScaleVec3F)(VemaVec3F v, const float s) {
	int i;
	for (i = 0; i < 3; ++i) {
		v[i] *= s;
	}
}

VEMA_IFC(void, ScaleSrcVec3F)(VemaVec3F v, const VemaVec3F vsrc, const float s) {
	int i;
	for (i = 0; i < 3; ++i) {
		v[i] = vsrc[i] * s;
	}
}

VEMA_IFC(float, DotVec3F)(const VemaVec3F v1, const VemaVec3F v2) {
	return v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2];
}

VEMA_IFC(float, TripleVec3F)(const VemaVec3F v0, const VemaVec3F v1, const VemaVec3F v2) {
	VemaVec3F cv;
	VEMA_FN(CrossVec3F)(cv, v0, v1);
	return VEMA_FN(DotVec3F)(cv, v2);
}

VEMA_IFC(float, MinElemVec3F)(const VemaVec3F v) {
	return VEMA_FN(Min3F)(v[0], v[1], v[2]);
}

VEMA_IFC(float, MaxElemVec3F)(const VemaVec3F v) {
	return VEMA_FN(Max3F)(v[0], v[1], v[2]);
}

VEMA_IFC(float, SqMagVec3F)(const VemaVec3F v) {
	float x = v[0];
	float y = v[1];
	float z = v[2];
	return x*x + y*y + z*z;
}

VEMA_IFC(float, FastMagVec3F)(const VemaVec3F v) {
	float x = v[0];
	float y = v[1];
	float z = v[2];
	return VEMA_FN(SqrtF)(x*x + y*y + z*z);
}

VEMA_IFC(float, MagVec3F)(const VemaVec3F v) {
	float x = v[0];
	float y = v[1];
	float z = v[2];
	float ax = VEMA_FN(AbsF)(x);
	float ay = VEMA_FN(AbsF)(y);
	float az = VEMA_FN(AbsF)(z);
	float ma = VEMA_FN(Max3F)(ax, ay, az);
	float mag = 0.0f;
	if (ma) {
		float s = 1.0f / ma;
		x *= s;
		y *= s;
		z *= s;
		mag = VEMA_FN(SqrtF)(x*x + y*y + z*z) * ma;
	}
	return mag;
}

VEMA_IFC(void, NormalizeVec3F)(VemaVec3F v) {
	float x = v[0];
	float y = v[1];
	float z = v[2];
	float ax = VEMA_FN(AbsF)(x);
	float ay = VEMA_FN(AbsF)(y);
	float az = VEMA_FN(AbsF)(z);
	float ma = VEMA_FN(Max3F)(ax, ay, az);
	if (ma > 0.0f) {
		float s = 1.0f / ma;
		x *= s;
		y *= s;
		z *= s;
		s = 1.0f / VEMA_FN(SqrtF)(x*x + y*y + z*z);
		x *= s;
		y *= s;
		z *= s;
		VEMA_FN(SetVec3F)(v, x, y, z);
	}
}

VEMA_IFC(void, NormalizeSrcVec3F)(VemaVec3F v, const VemaVec3F src) {
	VEMA_FN(CopyVec3F)(v, src);
	VEMA_FN(NormalizeVec3F)(v);
}

VEMA_IFC(void, NegVec3F)(VemaVec3F v) {
	int i;
	for (i = 0; i < 3; ++i) {
		v[i] = -v[i];
	}
}

VEMA_IFC(void, AbsVec3F)(VemaVec3F v) {
	int i;
	for (i = 0; i < 3; ++i) {
		v[i] = VEMA_FN(AbsF)(v[i]);
	}
}

VEMA_IFC(void, Rcp0Vec3F)(VemaVec3F v) {
	int i;
	for (i = 0; i < 3; ++i) {
		v[i] = VEMA_FN(Rcp0F)(v[i]);
	}
}

VEMA_IFC(void, LerpVec3F)(VemaVec3F v0, const VemaVec3F v1, const VemaVec3F v2, const float bias) {
	int i;
	float t = VEMA_FN(SaturateF)(bias);
	for (i = 0; i < 3; ++i) {
		v0[i] = VEMA_FN(LerpF)(v1[i], v2[i], t);
	}
}

VEMA_IFC(void, ClampVec3F)(VemaVec3F v, const VemaVec3F vmin, const VemaVec3F vmax) {
	int i;
	for (i = 0; i < 3; ++i) {
		v[i] = VEMA_FN(ClampF)(v[i], vmin[i], vmax[i]);
	}
}

VEMA_IFC(void, SaturateVec3F)(VemaVec3F v) {
	int i;
	for (i = 0; i < 3; ++i) {
		v[i] = VEMA_FN(SaturateF)(v[i]);
	}
}

VEMA_IFC(void, MinVec3F)(VemaVec3F v0, const VemaVec3F v1, const VemaVec3F v2) {
	int i;
	for (i = 0; i < 3; ++i) {
		v0[i] = VEMA_FN(MinF)(v1[i], v2[i]);
	}
}

VEMA_IFC(void, MaxVec3F)(VemaVec3F v0, const VemaVec3F v1, const VemaVec3F v2) {
	int i;
	for (i = 0; i < 3; ++i) {
		v0[i] = VEMA_FN(MaxF)(v1[i], v2[i]);
	}
}

VEMA_IFC(void, CombineVec3F)(VemaVec3F v0, const VemaVec3F v1, const float s1, const VemaVec3F v2, const float s2) {
	int i;
	for (i = 0; i < 3; ++i) {
		v0[i] = v1[i] * s1 + v2[i] * s2;
	}
}

VEMA_IFC(void, RotXYVec3F)(VemaVec3F v, const float rx, const float ry) {
	VemaQuatF q;
	VemaQuatF tq;
	VEMA_FN(RotYQuatF)(q, ry);
	VEMA_FN(RotXQuatF)(tq, rx);
	VEMA_FN(MulQuatF)(q, q, tq);
	VEMA_FN(ApplyQuatF)(v, v, q);
}

VEMA_IFC(int, AlmostEqualVec3F)(const VemaVec3F v1, const VemaVec3F v2, const float tol) {
	int i;
	for (i = 0; i < 3; ++i) {
		if (!VEMA_FN(AlmostEqualF)(v1[i], v2[i], tol)) {
			return 0;
		}
	}
	return 1;
}


VEMA_IFC(void, ZeroVec4F)(VemaVec4F v) {
	vemaMemSet(v, 0, sizeof(VemaVec4F));
}

VEMA_IFC(void, FillVec4F)(VemaVec4F v, const float s) {
	int i;
	for (i = 0; i < 4; ++i) {
		v[i] = s;
	}
}

VEMA_IFC(void, CopyVec4F)(VemaVec4F dst, const VemaVec4F src) {
	if (dst != src) {
		vemaMemCpy(dst, src, sizeof(VemaVec4F));
	}
}

VEMA_IFC(void, SetVec4F)(VemaVec4F v, const float x, const float y, const float z, const float w) {
	v[0] = x;
	v[1] = y;
	v[2] = z;
	v[3] = w;
}

VEMA_IFC(void, LoadVec4F)(VemaVec4F v, const void* pMem) {
	if (pMem) {
		vemaMemCpy(v, pMem, sizeof(VemaVec4F));
	}
}

VEMA_IFC(void, StoreVec4F)(void* pMem, const VemaVec4F v) {
	if (pMem) {
		vemaMemCpy(pMem, v, sizeof(VemaVec4F));
	}
}

VEMA_IFC(void, LoadAtIdxVec4F)(VemaVec4F v, const void* pMem, const int32_t idx) {
	if (pMem) {
		VEMA_FN(LoadVec4F)(v, (const VemaVec4F*)pMem + idx);
	}
}

VEMA_IFC(void, StoreAtIdxVec4F)(void* pMem, const int32_t idx, const VemaVec4F v) {
	if (pMem) {
		VEMA_FN(StoreVec4F)((VemaVec4F*)pMem + idx, v);
	}
}

VEMA_IFC(void, LoadAtOffsVec4F)(VemaVec4F v, const void* pMem, const size_t offs) {
	if (pMem) {
		VEMA_FN(LoadVec4F)(v, (const VemaVec4F*)((const uint8_t*)pMem + offs));
	}
}

VEMA_IFC(void, StoreAtOffsVec4F)(void* pMem, const size_t offs, const VemaVec4F v) {
	if (pMem) {
		VEMA_FN(StoreVec4F)((VemaVec4F*)((uint8_t*)pMem + offs), v);
	}
}

VEMA_IFC(void, ScaleVec4F)(VemaVec4F v, const float s) {
	int i;
	for (i = 0; i < 4; ++i) {
		v[i] *= s;
	}
}

VEMA_IFC(void, SaturateVec4F)(VemaVec4F v) {
	int i;
	for (i = 0; i < 4; ++i) {
		v[i] = VEMA_FN(SaturateF)(v[i]);
	}
}


VEMA_IFC(void, ZeroVec3D)(VemaVec3D v) {
	vemaMemSet(v, 0, sizeof(VemaVec3D));
}

VEMA_IFC(void, CopyVec3D)(VemaVec3D dst, const VemaVec3D src) {
	if (dst != src) {
		vemaMemCpy(dst, src, sizeof(VemaVec3D));
	}
}

VEMA_IFC(void, SetVec3D)(VemaVec3D v, const double x, const double y, const double z) {
	v[0] = x;
	v[1] = y;
	v[2] = z;
}

VEMA_IFC(void, AddVec3D)(VemaVec3D v0, const VemaVec3D v1, const VemaVec3D v2) {
	int i;
	for (i = 0; i < 3; ++i) {
		v0[i] = v1[i] + v2[i];
	}
}

VEMA_IFC(void, SubVec3D)(VemaVec3D v0, const VemaVec3D v1, const VemaVec3D v2) {
	int i;
	for (i = 0; i < 3; ++i) {
		v0[i] = v1[i] - v2[i];
	}
}

VEMA_IFC(void, MulVec3D)(VemaVec3D v0, const VemaVec3D v1, const VemaVec3D v2) {
	int i;
	for (i = 0; i < 3; ++i) {
		v0[i] = v1[i] * v2[i];
	}
}

VEMA_IFC(void, CrossVec3D)(VemaVec3D v0, const VemaVec3D v1, const VemaVec3D v2) {
	double v10 = v1[0];
	double v11 = v1[1];
	double v12 = v1[2];
	double v20 = v2[0];
	double v21 = v2[1];
	double v22 = v2[2];
	v0[0] = v11*v22 - v12*v21;
	v0[1] = v12*v20 - v10*v22;
	v0[2] = v10*v21 - v11*v20;
}

VEMA_IFC(void, ScaleVec3D)(VemaVec3D v, const double s) {
	int i;
	for (i = 0; i < 3; ++i) {
		v[i] *= s;
	}
}

VEMA_IFC(void, ScaleSrcVec3D)(VemaVec3D v, const VemaVec3D vsrc, const double s) {
	int i;
	for (i = 0; i < 3; ++i) {
		v[i] = vsrc[i] * s;
	}
}

VEMA_IFC(double, DotVec3D)(const VemaVec3D v1, const VemaVec3D v2) {
	return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
}

VEMA_IFC(double, TripleVec3D)(const VemaVec3D v0, const VemaVec3D v1, const VemaVec3D v2) {
	VemaVec3D cv;
	VEMA_FN(CrossVec3D)(cv, v0, v1);
	return VEMA_FN(DotVec3D)(cv, v2);
}

VEMA_IFC(double, SqMagVec3D)(const VemaVec3D v) {
	double x = v[0];
	double y = v[1];
	double z = v[2];
	return x*x + y*y + z*z;
}

VEMA_IFC(double, FastMagVec3D)(const VemaVec3D v) {
	double x = v[0];
	double y = v[1];
	double z = v[2];
	return VEMA_FN(SqrtD)(x*x + y*y + z*z);
}

VEMA_IFC(double, MagVec3D)(const VemaVec3D v) {
	double x = v[0];
	double y = v[1];
	double z = v[2];
	double ax = VEMA_FN(AbsD)(x);
	double ay = VEMA_FN(AbsD)(y);
	double az = VEMA_FN(AbsD)(z);
	double ma = VEMA_FN(Max3D)(ax, ay, az);
	double mag = 0.0;
	if (ma) {
		double s = 1.0 / ma;
		x *= s;
		y *= s;
		z *= s;
		mag = VEMA_FN(SqrtD)(x*x + y*y + z*z) * ma;
	}
	return mag;
}

VEMA_IFC(void, NormalizeVec3D)(VemaVec3D v) {
	double x = v[0];
	double y = v[1];
	double z = v[2];
	double ax = VEMA_FN(AbsD)(x);
	double ay = VEMA_FN(AbsD)(y);
	double az = VEMA_FN(AbsD)(z);
	double ma = VEMA_FN(Max3D)(ax, ay, az);
	if (ma > 0.0) {
		double s = 1.0 / ma;
		x *= s;
		y *= s;
		z *= s;
		s = 1.0 / VEMA_FN(SqrtD)(x*x + y*y + z*z);
		x *= s;
		y *= s;
		z *= s;
		VEMA_FN(SetVec3D)(v, x, y, z);
	}
}

VEMA_IFC(void, NormalizeSrcVec3D)(VemaVec3D v, const VemaVec3D src) {
	VEMA_FN(CopyVec3D)(v, src);
	VEMA_FN(NormalizeVec3D)(v);
}


VEMA_IFC(void, ZeroMtx4x4F)(VemaMtx4x4F m) {
	vemaMemSet(m, 0, sizeof(VemaMtx4x4F));
}

VEMA_IFC(void, IdentityMtx4x4F)(VemaMtx4x4F m) {
	static VemaMtx4x4F im = {
		{ 1.0f, 0.0f, 0.0f, 0.0f },
		{ 0.0f, 1.0f, 0.0f, 0.0f },
		{ 0.0f, 0.0f, 1.0f, 0.0f },
		{ 0.0f, 0.0f, 0.0f, 1.0f }
	};
	VEMA_FN(CopyMtx4x4F)(m, im);
}

VEMA_IFC(void, CopyMtx4x4F)(VemaMtx4x4F dst, const VemaMtx4x4F src) {
	if (dst != src) {
		vemaMemCpy(dst, src, sizeof(VemaMtx4x4F));
	}
}

VEMA_IFC(void, LoadMtx4x4F)(VemaMtx4x4F m, const void* pMem) {
	if (pMem) {
		vemaMemCpy(m, pMem, sizeof(VemaMtx4x4F));
	}
}

VEMA_IFC(void, StoreMtx4x4F)(void* pMem, const VemaMtx4x4F m) {
	if (pMem) {
		vemaMemCpy(pMem, m, sizeof(VemaMtx4x4F));
	}
}

VEMA_IFC(void, LoadAtIdxMtx4x4F)(VemaMtx4x4F m, const void* pMem, const int32_t idx) {
	if (pMem) {
		VEMA_FN(LoadMtx4x4F)(m, (const VemaMtx4x4F*)pMem + idx);
	}
}

VEMA_IFC(void, StoreAtIdxMtx4x4F)(void* pMem, const int32_t idx, const VemaMtx4x4F m) {
	if (pMem) {
		VEMA_FN(StoreMtx4x4F)((VemaMtx4x4F*)pMem + idx, m);
	}
}

VEMA_IFC(void, LoadAtOffsMtx4x4F)(VemaMtx4x4F m, const void* pMem, const size_t offs) {
	if (pMem) {
		VEMA_FN(LoadMtx4x4F)(m, (const VemaMtx4x4F*)((const uint8_t*)pMem + offs));
	}
}

VEMA_IFC(void, StoreAtOffsMtx4x4F)(void* pMem, const size_t offs, const VemaMtx4x4F m) {
	if (pMem) {
		VEMA_FN(StoreMtx4x4F)((VemaMtx4x4F*)((uint8_t*)pMem + offs), m);
	}
}

VEMA_IFC(void, MulMtx4x4F)(VemaMtx4x4F m0, const VemaMtx4x4F m1, const VemaMtx4x4F m2) {
#ifdef VEMA_AVX
	float* pD = (float*)m0;
	float* pRA = (float*)m1;
	__m128* pRB = (__m128*)m2;
	__m256 rA0A1 = _mm256_loadu_ps(&pRA[0]);
	__m256 rA2A3 = _mm256_loadu_ps(&pRA[8]);
	__m256 rB0 = _mm256_broadcast_ps(&pRB[0]);
	__m256 rB1 = _mm256_broadcast_ps(&pRB[1]);
	__m256 rB2 = _mm256_broadcast_ps(&pRB[2]);
	__m256 rB3 = _mm256_broadcast_ps(&pRB[3]);
	_mm256_storeu_ps(&pD[0], _mm256_add_ps(
		_mm256_add_ps(
			_mm256_mul_ps(_mm256_shuffle_ps(rA0A1, rA0A1, VEMA_SIMD_ELEM_MASK(0)), rB0),
			_mm256_mul_ps(_mm256_shuffle_ps(rA0A1, rA0A1, VEMA_SIMD_ELEM_MASK(1)), rB1)
			),
		_mm256_add_ps(
			_mm256_mul_ps(_mm256_shuffle_ps(rA0A1, rA0A1, VEMA_SIMD_ELEM_MASK(2)), rB2),
			_mm256_mul_ps(_mm256_shuffle_ps(rA0A1, rA0A1, VEMA_SIMD_ELEM_MASK(3)), rB3)
			)
		)
	);
	_mm256_storeu_ps(&pD[8], _mm256_add_ps(
		_mm256_add_ps(
			_mm256_mul_ps(_mm256_shuffle_ps(rA2A3, rA2A3, VEMA_SIMD_ELEM_MASK(0)), rB0),
			_mm256_mul_ps(_mm256_shuffle_ps(rA2A3, rA2A3, VEMA_SIMD_ELEM_MASK(1)), rB1)
			),
		_mm256_add_ps(
			_mm256_mul_ps(_mm256_shuffle_ps(rA2A3, rA2A3, VEMA_SIMD_ELEM_MASK(2)), rB2),
			_mm256_mul_ps(_mm256_shuffle_ps(rA2A3, rA2A3, VEMA_SIMD_ELEM_MASK(3)), rB3)
			)
		)
	);
#elif defined(VEMA_NEON)
	float* p = (float*)m0;
	float* pA = (float*)m1;
	float* pB = (float*)m2;
	float32x4_t rA0 = vld1q_f32(&pA[0x0]);
	float32x4_t rA1 = vld1q_f32(&pA[0x4]);
	float32x4_t rA2 = vld1q_f32(&pA[0x8]);
	float32x4_t rA3 = vld1q_f32(&pA[0xC]);
	float32x4_t rB0 = vld1q_f32(&pB[0x0]);
	float32x4_t rB1 = vld1q_f32(&pB[0x4]);
	float32x4_t rB2 = vld1q_f32(&pB[0x8]);
	float32x4_t rB3 = vld1q_f32(&pB[0xC]);
	vst1q_f32(&p[0x0],
		vaddq_f32(vaddq_f32(vaddq_f32(
			vmulq_f32(vemaNeonElem(rA0, 0), rB0), vmulq_f32(vemaNeonElem(rA0, 1), rB1)),
			vmulq_f32(vemaNeonElem(rA0, 2), rB2)), vmulq_f32(vemaNeonElem(rA0, 3), rB3)));
	vst1q_f32(&p[0x4],
		vaddq_f32(vaddq_f32(vaddq_f32(
			vmulq_f32(vemaNeonElem(rA1, 0), rB0), vmulq_f32(vemaNeonElem(rA1, 1), rB1)),
			vmulq_f32(vemaNeonElem(rA1, 2), rB2)), vmulq_f32(vemaNeonElem(rA1, 3), rB3)));
	vst1q_f32(&p[0x8],
		vaddq_f32(vaddq_f32(vaddq_f32(
			vmulq_f32(vemaNeonElem(rA2, 0), rB0), vmulq_f32(vemaNeonElem(rA2, 1), rB1)),
			vmulq_f32(vemaNeonElem(rA2, 2), rB2)), vmulq_f32(vemaNeonElem(rA2, 3), rB3)));
	vst1q_f32(&p[0xC],
		vaddq_f32(vaddq_f32(vaddq_f32(
			vmulq_f32(vemaNeonElem(rA3, 0), rB0), vmulq_f32(vemaNeonElem(rA3, 1), rB1)),
			vmulq_f32(vemaNeonElem(rA3, 2), rB2)), vmulq_f32(vemaNeonElem(rA3, 3), rB3)));
#else
	VemaMtx4x4F tm;
	VEMA_FN(MulMtxF)(&tm[0][0], &m1[0][0], &m2[0][0], 4, 4, 4);
	VEMA_FN(CopyMtx4x4F)(m0, tm);
#endif
}

VEMA_IFC(void, MulAryMtx4x4F)(VemaMtx4x4F* pDst, const VemaMtx4x4F* pSrc1, const VemaMtx4x4F* pSrc2, const size_t n) {
	size_t i;
	for (i = 0; i < n; ++i) {
		VEMA_FN(MulMtx4x4F)(pDst[i], pSrc1[i], pSrc2[i]);
	}
}

VEMA_IFC(void, TransposeMtx4x4F)(VemaMtx4x4F m) {
	float t;
	t = m[0][1]; m[0][1] = m[1][0]; m[1][0] = t;
	t = m[0][2]; m[0][2] = m[2][0]; m[2][0] = t;
	t = m[0][3]; m[0][3] = m[3][0]; m[3][0] = t;
	t = m[1][2]; m[1][2] = m[2][1]; m[2][1] = t;
	t = m[1][3]; m[1][3] = m[3][1]; m[3][1] = t;
	t = m[2][3]; m[2][3] = m[3][2]; m[3][2] = t;
}

VEMA_IFC(void, TransposeSrcMtx4x4F)(VemaMtx4x4F m, const VemaMtx4x4F src) {
	float t;
	m[0][0] = src[0][0];
	m[1][1] = src[1][1];
	m[2][2] = src[2][2];
	m[3][3] = src[3][3];
	t = src[0][1]; m[0][1] = src[1][0]; m[1][0] = t;
	t = src[0][2]; m[0][2] = src[2][0]; m[2][0] = t;
	t = src[0][3]; m[0][3] = src[3][0]; m[3][0] = t;
	t = src[1][2]; m[1][2] = src[2][1]; m[2][1] = t;
	t = src[1][3]; m[1][3] = src[3][1]; m[3][1] = t;
	t = src[2][3]; m[2][3] = src[3][2]; m[3][2] = t;
}

VEMA_IFC(void, TransposeAxesMtx4x4F)(VemaMtx4x4F m) {
	float t;
	t = m[0][1]; m[0][1] = m[1][0]; m[1][0] = t;
	t = m[0][2]; m[0][2] = m[2][0]; m[2][0] = t;
	t = m[1][2]; m[1][2] = m[2][1]; m[2][1] = t;
}

VEMA_IFC(void, InvertMtx4x4F)(VemaMtx4x4F m) {
	int wk[4 * 3];
	int ok = VEMA_FN(GJInvertMtxF)(&m[0][0], 4, wk);
	if (!ok) {
		VEMA_FN(ZeroMtx4x4F)(m);
	}
}

VEMA_IFC(void, InvertSrcMtx4x4F)(VemaMtx4x4F m, const VemaMtx4x4F src) {
	VEMA_FN(CopyMtx4x4F)(m, src);
	VEMA_FN(InvertMtx4x4F)(m);
}

VEMA_IFC(void, NrmAxisRotMtx4x4F)(VemaMtx4x4F m, const VemaVec3F v, const float ang) {
	VemaVec3F vv;
	float s;
	float c;
	float t;
	float x;
	float y;
	float z;
	float xx;
	float yy;
	float zz;
	float xy;
	float xz;
	float yz;
	VEMA_FN(MulVec3F)(vv, v, v);
	s = VEMA_FN(SinF)(ang);
	c = VEMA_FN(CosF)(ang);
	t = 1.0f - c;
	x = v[0];
	y = v[1];
	z = v[2];
	xx = vv[0];
	yy = vv[1];
	zz = vv[2];
	xy = x * y;
	xz = x * z;
	yz = y * z;
	VEMA_FN(AxisXValsMtx4x4F)(m, t*xx + c, t*xy + s*z, t*xz - s*y);
	VEMA_FN(AxisYValsMtx4x4F)(m, t*xy - s*z, t*yy + c, t*yz + s*x);
	VEMA_FN(AxisZValsMtx4x4F)(m, t*xz + s*y, t*yz - s*x, t*zz + c);
	VEMA_FN(ZeroTranslationMtx4x4F)(m);
}

VEMA_IFC(void, AxisRotMtx4x4F)(VemaMtx4x4F m, const VemaVec3F axis, const float ang) {
	VemaVec3F v;
	VEMA_FN(NormalizeSrcVec3F)(v, axis);
	VEMA_FN(NrmAxisRotMtx4x4F)(m, v, ang);
}

VEMA_IFC(void, RotXMtx4x4F)(VemaMtx4x4F m, const float ang) {
	VemaVec3F v;
	VEMA_FN(SetVec3F)(v, 1.0f, 0.0f, 0.0f);
	VEMA_FN(NrmAxisRotMtx4x4F)(m, v, ang);
}

VEMA_IFC(void, RotYMtx4x4F)(VemaMtx4x4F m, const float ang) {
	VemaVec3F v;
	VEMA_FN(SetVec3F)(v, 0.0f, 1.0f, 0.0f);
	VEMA_FN(NrmAxisRotMtx4x4F)(m, v, ang);
}

VEMA_IFC(void, RotZMtx4x4F)(VemaMtx4x4F m, const float ang) {
	VemaVec3F v;
	VEMA_FN(SetVec3F)(v, 0.0f, 0.0f, 1.0f);
	VEMA_FN(NrmAxisRotMtx4x4F)(m, v, ang);
}

VEMA_IFC(void, AxisXMtx4x4F)(VemaMtx4x4F m, const VemaVec3F v) {
	VEMA_FN(AxisXValsMtx4x4F)(m, v[0], v[1], v[2]);
}

VEMA_IFC(void, AxisXValsMtx4x4F)(VemaMtx4x4F m, const float x, const float y, const float z) {
	m[0][0] = x;
	m[0][1] = y;
	m[0][2] = z;
	m[0][3] = 0.0f;
}

VEMA_IFC(void, AxisYMtx4x4F)(VemaMtx4x4F m, const VemaVec3F v) {
	VEMA_FN(AxisYValsMtx4x4F)(m, v[0], v[1], v[2]);
}

VEMA_IFC(void, AxisYValsMtx4x4F)(VemaMtx4x4F m, const float x, const float y, const float z) {
	m[1][0] = x;
	m[1][1] = y;
	m[1][2] = z;
	m[1][3] = 0.0f;
}

VEMA_IFC(void, AxisZMtx4x4F)(VemaMtx4x4F m, const VemaVec3F v) {
	VEMA_FN(AxisZValsMtx4x4F)(m, v[0], v[1], v[2]);
}

VEMA_IFC(void, AxisZValsMtx4x4F)(VemaMtx4x4F m, const float x, const float y, const float z) {
	m[2][0] = x;
	m[2][1] = y;
	m[2][2] = z;
	m[2][3] = 0.0f;
}

VEMA_IFC(void, QuatAxesMtx4x4F)(VemaMtx4x4F m, const VemaQuatF q) {
	VemaVec3F ax;
	VemaVec3F ay;
	VemaVec3F az;
	VEMA_FN(GetAxisXQuatF)(ax, q);
	VEMA_FN(GetAxisYQuatF)(ay, q);
	VEMA_FN(GetAxisZQuatF)(az, q);
	VEMA_FN(AxisXMtx4x4F)(m, ax);
	VEMA_FN(AxisYMtx4x4F)(m, ay);
	VEMA_FN(AxisZMtx4x4F)(m, az);
}

VEMA_IFC(void, TranslationMtx4x4F)(VemaMtx4x4F m, const VemaVec3F t) {
	VEMA_FN(TranslationValsMtx4x4F)(m, t[0], t[1], t[2]);
}

VEMA_IFC(void, TranslationValsMtx4x4F)(VemaMtx4x4F m, const float tx, const float ty, const float tz) {
	m[3][0] = tx;
	m[3][1] = ty;
	m[3][2] = tz;
	m[3][3] = 1.0f;
}

VEMA_IFC(void, ZeroTranslationMtx4x4F)(VemaMtx4x4F m) {
	m[3][0] = 0.0f;
	m[3][1] = 0.0f;
	m[3][2] = 0.0f;
	m[3][3] = 1.0f;
}

VEMA_IFC(void, GetAxisXMtx4x4F)(VemaVec3F v, const VemaMtx4x4F m) {
	vemaMemCpy(v, m[0], sizeof(VemaVec3F));
}

VEMA_IFC(void, GetAxisYMtx4x4F)(VemaVec3F v, const VemaMtx4x4F m) {
	vemaMemCpy(v, m[1], sizeof(VemaVec3F));
}

VEMA_IFC(void, GetAxisZMtx4x4F)(VemaVec3F v, const VemaMtx4x4F m) {
	vemaMemCpy(v, m[2], sizeof(VemaVec3F));
}

VEMA_IFC(void, GetTranslationMtx4x4F)(VemaVec3F v, const VemaMtx4x4F m) {
	vemaMemCpy(v, m[3], sizeof(VemaVec3F));
}

VEMA_IFC(void, ScalingMtx4x4F)(VemaMtx4x4F m, const float sx, const float sy, const float sz) {
	VEMA_FN(IdentityMtx4x4F)(m);
	m[0][0] = sx;
	m[1][1] = sy;
	m[2][2] = sz;
}

VEMA_IFC(void, UniformScalingMtx4x4F)(VemaMtx4x4F m, const float s) {
	VEMA_FN(ScalingMtx4x4F)(m, s, s, s);
}

VEMA_IFC(void, ViewMtx4x4F)(VemaMtx4x4F m, const VemaVec3F pos, const VemaVec3F tgt, const VemaVec3F up, VemaMtx4x4F* pInv) {
	VemaVec3F dir;
	VemaVec3F side;
	VemaVec3F upvec;
	VemaVec3F org;
	VEMA_FN(SubVec3F)(dir, tgt, pos);
	VEMA_FN(NormalizeVec3F)(dir);
	VEMA_FN(CrossVec3F)(side, up, dir);
	VEMA_FN(NormalizeVec3F)(side);
	VEMA_FN(CrossVec3F)(upvec, side, dir);
	VEMA_FN(NegVec3F)(dir);
	VEMA_FN(NegVec3F)(side);
	VEMA_FN(NegVec3F)(upvec);
	if (pInv) {
		VEMA_FN(AxisXMtx4x4F)(*pInv, side);
		VEMA_FN(AxisYMtx4x4F)(*pInv, upvec);
		VEMA_FN(AxisZMtx4x4F)(*pInv, dir);
		VEMA_FN(TranslationMtx4x4F)(*pInv, pos);
	}
	VEMA_FN(AxisXMtx4x4F)(m, side);
	VEMA_FN(AxisYMtx4x4F)(m, upvec);
	VEMA_FN(AxisZMtx4x4F)(m, dir);
	VEMA_FN(ZeroTranslationMtx4x4F)(m);
	VEMA_FN(TransposeAxesMtx4x4F)(m);
	VEMA_FN(CopyVec3F)(org, pos);
	VEMA_FN(NegVec3F)(org);
	VEMA_FN(CalcPntMtx4x4F)(org, org, m);
	VEMA_FN(TranslationMtx4x4F)(m, org);
}

VEMA_IFC(void, PerspectiveMtx4x4F)(VemaMtx4x4F m, const float fovy, const float aspect, const float znear, const float zfar) {
	float h = fovy * 0.5f;
	float s = vemaSinF(h);
	float c = vemaCosF(h);
	float cot = c / s;
	float q = zfar / (zfar - znear);
	VEMA_FN(ZeroMtx4x4F)(m);
	m[2][3] = -1.0f;
	m[0][0] = cot / aspect;
	m[1][1] = cot;
	m[2][2] = -q;
	m[3][2] = -q * znear;
}

VEMA_IFC(void, CalcVecMtx4x4F)(VemaVec3F dst, const VemaVec3F src, const VemaMtx4x4F m) {
	float res[4];
	VEMA_FN(MulVecMtxF)(res, src, &m[0][0], 3, 4);
	VEMA_FN(SetVec3F)(dst, res[0], res[1], res[2]);
}

VEMA_IFC(void, CalcPntMtx4x4F)(VemaVec3F dst, const VemaVec3F src, const VemaMtx4x4F m) {
	float pnt[4];
	float res[4];
	pnt[0] = src[0];
	pnt[1] = src[1];
	pnt[2] = src[2];
	pnt[3] = 1.0f;
	VEMA_FN(MulVecMtxF)(res, pnt, &m[0][0], 4, 4);
	VEMA_FN(SetVec3F)(dst, res[0], res[1], res[2]);
}

VEMA_IFC(void, FromQuatMtx4x4F)(VemaMtx4x4F m, const VemaQuatF q) {
	VEMA_FN(QuatAxesMtx4x4F)(m, q);
	VEMA_FN(ZeroTranslationMtx4x4F)(m);
}


VEMA_IFC(void, ZeroMtx3x4F)(VemaMtx3x4F m) {
	vemaMemSet(m, 0, sizeof(VemaMtx3x4F));
}

VEMA_IFC(void, IdentityMtx3x4F)(VemaMtx3x4F m) {
	static VemaMtx3x4F im = {
		{ 1.0f, 0.0f, 0.0f, 0.0f },
		{ 0.0f, 1.0f, 0.0f, 0.0f },
		{ 0.0f, 0.0f, 1.0f, 0.0f },
	};
	VEMA_FN(CopyMtx3x4F)(m, im);
}

VEMA_IFC(void, CopyMtx3x4F)(VemaMtx3x4F dst, const VemaMtx3x4F src) {
	if (dst != src) {
		vemaMemCpy(dst, src, sizeof(VemaMtx3x4F));
	}
}

VEMA_IFC(void, ConvertMtx4x4FtoMtx3x4F)(VemaMtx3x4F dst, const VemaMtx4x4F src) {
	VemaMtx4x4F tm;
	VEMA_FN(TransposeSrcMtx4x4F)(tm, src);
	vemaMemCpy(dst, tm, sizeof(VemaMtx3x4F));
}

VEMA_IFC(void, ConcatMtx3x4F)(VemaMtx3x4F m0, const VemaMtx3x4F m1, const VemaMtx3x4F m2) {
	float a00 = m1[0][0]; float a01 = m1[1][0]; float a02 = m1[2][0];
	float a10 = m1[0][1]; float a11 = m1[1][1]; float a12 = m1[2][1];
	float a20 = m1[0][2]; float a21 = m1[1][2]; float a22 = m1[2][2];
	float a30 = m1[0][3]; float a31 = m1[1][3]; float a32 = m1[2][3];
	float b00 = m2[0][0]; float b01 = m2[1][0]; float b02 = m2[2][0];
	float b10 = m2[0][1]; float b11 = m2[1][1]; float b12 = m2[2][1];
	float b20 = m2[0][2]; float b21 = m2[1][2]; float b22 = m2[2][2];
	float b30 = m2[0][3]; float b31 = m2[1][3]; float b32 = m2[2][3];
	m0[0][0] = a00*b00 + a01*b10 + a02*b20;
	m0[0][1] = a10*b00 + a11*b10 + a12*b20;
	m0[0][2] = a20*b00 + a21*b10 + a22*b20;
	m0[0][3] = a30*b00 + a31*b10 + a32*b20 + b30;
	m0[1][0] = a00*b01 + a01*b11 + a02*b21;
	m0[1][1] = a10*b01 + a11*b11 + a12*b21;
	m0[1][2] = a20*b01 + a21*b11 + a22*b21;
	m0[1][3] = a30*b01 + a31*b11 + a32*b21 + b31;
	m0[2][0] = a00*b02 + a01*b12 + a02*b22;
	m0[2][1] = a10*b02 + a11*b12 + a12*b22;
	m0[2][2] = a20*b02 + a21*b12 + a22*b22;
	m0[2][3] = a30*b02 + a31*b12 + a32*b22 + b32;
}


VEMA_IFC(void, IdentityQuatF)(VemaQuatF q) {
	VEMA_FN(SetQuatF)(q, 0.0f, 0.0f, 0.0f, 1.0f);
}

VEMA_IFC(void, CopyQuatF)(VemaQuatF dst, const VemaQuatF src) {
	if (dst != src) {
		vemaMemCpy(dst, src, sizeof(VemaQuatF));
	}
}

VEMA_IFC(void, SetQuatF)(VemaQuatF q, const float x, const float y, const float z, const float w) {
	q[0] = x;
	q[1] = y;
	q[2] = z;
	q[3] = w;
}

VEMA_IFC(void, SetPartsQuatF)(VemaQuatF q, const VemaVec3F im, const float re) {
	q[0] = im[0];
	q[1] = im[1];
	q[2] = im[2];
	q[3] = re;
}

VEMA_IFC(void, AxisRotQuatF)(VemaQuatF q, const VemaVec3F axis, const float ang) {
	VemaVec3F v;
	VEMA_FN(NormalizeSrcVec3F)(v, axis);
	float h = ang * 0.5f;
	float s = VEMA_FN(SinF)(h);
	float c = VEMA_FN(CosF)(h);
	VEMA_FN(ScaleVec3F)(v, s);
	VEMA_FN(SetPartsQuatF)(q, v, c);
}

VEMA_IFC(void, RotXQuatF)(VemaQuatF q, const float ang) {
	float h = ang * 0.5f;
	float s = VEMA_FN(SinF)(h);
	float c = VEMA_FN(CosF)(h);
	VEMA_FN(SetQuatF)(q, s, 0.0f, 0.0f, c);
}

VEMA_IFC(void, RotYQuatF)(VemaQuatF q, const float ang) {
	float h = ang * 0.5f;
	float s = VEMA_FN(SinF)(h);
	float c = VEMA_FN(CosF)(h);
	VEMA_FN(SetQuatF)(q, 0.0f, s, 0.0f, c);
}

VEMA_IFC(void, RotZQuatF)(VemaQuatF q, const float ang) {
	float h = ang * 0.5f;
	float s = VEMA_FN(SinF)(h);
	float c = VEMA_FN(CosF)(h);
	VEMA_FN(SetQuatF)(q, 0.0f, 0.0f, s, c);
}

VEMA_IFC(void, GetAxisXQuatF)(VemaVec3F v, const VemaQuatF q) {
	float x = q[0];
	float y = q[1];
	float z = q[2];
	float w = q[3];
	v[0] = 1.0f - (2.0f*y*y) - (2.0f*z*z);
	v[1] = (2.0f*x*y) + (2.0f*w*z);
	v[2] = (2.0f*x*z) - (2.0f*w*y);
}

VEMA_IFC(void, GetAxisYQuatF)(VemaVec3F v, const VemaQuatF q) {
	float x = q[0];
	float y = q[1];
	float z = q[2];
	float w = q[3];
	v[0] = (2.0f*x*y) - (2.0f*w*z);
	v[1] = 1.0f - (2.0f*x*x) - (2.0f*z*z);
	v[2] = (2.0f*y*z) + (2.0f*w*x);
}

VEMA_IFC(void, GetAxisZQuatF)(VemaVec3F v, const VemaQuatF q) {
	float x = q[0];
	float y = q[1];
	float z = q[2];
	float w = q[3];
	v[0] = (2.0f*x*z) + (2.0f*w*y);
	v[1] = (2.0f*y*z) - (2.0f*w*x);
	v[2] = 1.0f - (2.0f*x*x) - (2.0f*y*y);
}

VEMA_IFC(void, MulQuatF)(VemaQuatF q0, const VemaQuatF q1, const VemaQuatF q2) {
	float q1x = q1[0];
	float q1y = q1[1];
	float q1z = q1[2];
	float q1w = q1[3];
	float q2x = q2[0];
	float q2y = q2[1];
	float q2z = q2[2];
	float q2w = q2[3];
	float x = q1w*q2x + q1x*q2w + q1y*q2z - q1z*q2y;
	float y = q1w*q2y + q1y*q2w + q1z*q2x - q1x*q2z;
	float z = q1w*q2z + q1z*q2w + q1x*q2y - q1y*q2x;
	float w = q1w*q2w - q1x*q2x - q1y*q2y - q1z*q2z;
	VEMA_FN(SetQuatF)(q0, x, y, z, w);
}

VEMA_IFC(void, SlerpQuatF)(VemaQuatF q, const VemaQuatF q1, const VemaQuatF q2, const float bias) {
	VemaQuatF tq1;
	VemaQuatF tq2;
	int i;
	float ang;
	float s;
	float d;
	float t = VEMA_FN(SaturateF)(bias);
	float u = 0.0f;
	float v = 0.0f;
	VEMA_FN(CopyQuatF)(tq1, q1);
	VEMA_FN(CopyQuatF)(tq2, q2);
	if (VEMA_FN(DotQuatF)(q1, q2) < 0.0f) {
		VEMA_FN(NegQuatF)(tq2);
	}
	for (i = 0; i < 4; ++i) {
		u += VEMA_FN(SqF)(tq1[i] - tq2[i]);
		v += VEMA_FN(SqF)(tq1[i] + tq2[i]);
	}
	ang = 2.0f * VEMA_FN(ArcTan2F)(VEMA_FN(SqrtF)(u), VEMA_FN(SqrtF)(v));
	s = 1.0f - t;
	d = VEMA_FN(InvSincF)(ang);
	s = VEMA_FN(SincF)(ang*s) * d * s;
	t = VEMA_FN(SincF)(ang*t) * d * t;
	for (i = 0; i < 4; ++i) {
		q[i] = tq1[i]*s + tq2[i]*t;
	}
	VEMA_FN(NormalizeQuatF)(q);
}

VEMA_IFC(void, ScaleQuatF)(VemaQuatF q, const float s) {
	int i;
	for (i = 0; i < 4; ++i) {
		q[i] *= s;
	}
}

VEMA_IFC(void, NormalizeQuatF)(VemaQuatF q) {
	float x = q[0];
	float y = q[1];
	float z = q[2];
	float w = q[3];
	float ax = VEMA_FN(AbsF)(x);
	float ay = VEMA_FN(AbsF)(y);
	float az = VEMA_FN(AbsF)(z);
	float aw = VEMA_FN(AbsF)(w);
	float ma = VEMA_FN(Max4F)(ax, ay, az, aw);
	if (ma > 0.0f) {
		float s = 1.0f / ma;
		VEMA_FN(ScaleQuatF)(q, s);
		x = q[0];
		y = q[1];
		z = q[2];
		w = q[3];
		s = 1.0f / VEMA_FN(SqrtF)(x*x + y*y + z*z + w*w);
		VEMA_FN(ScaleQuatF)(q, s);
	}
}

VEMA_IFC(void, NegQuatF)(VemaQuatF q) {
	int i;
	for (i = 0; i < 4; ++i) {
		q[i] = -q[i];
	}
}

VEMA_IFC(void, ConjugateQuatF)(VemaQuatF q) {
	int i;
	for (i = 0; i < 3; ++i) {
		q[i] = -q[i];
	}
}

VEMA_IFC(float, DotQuatF)(const VemaQuatF q1, const VemaQuatF q2) {
	return q1[0]*q2[0] + q1[1]*q2[1] + q1[2]*q2[2] + q1[3]*q2[3];
}

VEMA_IFC(float, RePartQuatF)(const VemaQuatF q) {
	return q[3];
}

VEMA_IFC(void, ImPartQuatF)(VemaVec3F v, const VemaQuatF q) {
	int i;
	for (i = 0; i < 3; ++i) {
		v[i] = q[i];
	}
}

VEMA_IFC(void, FromMtx4x4QuatF)(VemaQuatF q, const VemaMtx4x4F m) {
	float s;
	float x = 0.0f;
	float y = 0.0f;
	float z = 0.0f;
	float w = 1.0f;
	float trace = m[0][0] + m[1][1] + m[2][2];
	if (trace > 0.0f) {
		s = VEMA_FN(SqrtF)(trace + 1.0f);
		w = s * 0.5f;
		s = 0.5f / s;
		x = (m[1][2] - m[2][1]) * s;
		y = (m[2][0] - m[0][2]) * s;
		z = (m[0][1] - m[1][0]) * s;
	} else {
		if (m[1][1] > m[0][0]) {
			if (m[2][2] > m[1][1]) {
				s = m[2][2] - m[1][1] - m[0][0];
				s = VEMA_FN(SqrtF)(s + 1.0f);
				z = s * 0.5f;
				if (s != 0.0f) {
					s = 0.5f / s;
				}
				w = (m[0][1] - m[1][0]) * s;
				x = (m[2][0] + m[0][2]) * s;
				y = (m[2][1] + m[1][2]) * s;
			} else {
				s = m[1][1] - m[2][2] - m[0][0];
				s = VEMA_FN(SqrtF)(s + 1.0f);
				y = s * 0.5f;
				if (s != 0.0f) {
					s = 0.5f / s;
				}
				w = (m[2][0] - m[0][2]) * s;
				z = (m[1][2] + m[2][1]) * s;
				x = (m[1][0] + m[0][1]) * s;
			}
		} else if (m[2][2] > m[0][0]) {
			s = m[2][2] - m[1][1] - m[0][0];
			s = VEMA_FN(SqrtF)(s + 1.0f);
			z = s * 0.5f;
			if (s != 0.0f) {
				s = 0.5f / s;
			}
			w = (m[0][1] - m[1][0]) * s;
			x = (m[2][0] + m[0][2]) * s;
			y = (m[2][1] + m[1][2]) * s;
		} else {
			s = m[0][0] - m[1][1] - m[2][2];
			s = VEMA_FN(SqrtF)(s + 1.0f);
			x = s * 0.5f;
			if (s != 0.0f) {
				s = 0.5f / s;
			}
			w = (m[1][2] - m[2][1]) * s;
			y = (m[0][1] + m[1][0]) * s;
			z = (m[0][2] + m[2][0]) * s;
		}
	}
	VEMA_FN(SetQuatF)(q, x, y, z, w);
}

VEMA_IFC(void, ApplyQuatF)(VemaVec3F dst, const VemaVec3F src, const VemaQuatF q) {
	float d;
	float qs;
	float qss;
	VemaVec3F qv;
	VemaVec3F tv1;
	VemaVec3F tv2;
	VEMA_FN(ImPartQuatF)(qv, q);
	qs = VEMA_FN(RePartQuatF)(q);
	qss = VEMA_FN(SqF)(qs);
	d = VEMA_FN(DotVec3F)(src, qv);
	VEMA_FN(CombineVec3F)(tv1, qv, d, src, qss);
	VEMA_FN(CrossVec3F)(tv2, src, qv);
	VEMA_FN(ScaleVec3F)(tv2, qs);
	VEMA_FN(SubVec3F)(tv1, tv1, tv2);
	VEMA_FN(ScaleVec3F)(tv1, 2.0f);
	VEMA_FN(SubVec3F)(dst, tv1, src);
}

VEMA_IFC(float, FastMagQuatF)(const VemaQuatF q) {
	int i;
	float d = 0.0f;
	for (i = 0; i < 4; ++i) {
		d += q[i] * q[i];
	}
	return VEMA_FN(SqrtF)(d);
}

VEMA_IFC(float, HalfAngQuatF)(const VemaQuatF q) {
	return VEMA_FN(ArcCosF)(VEMA_FN(RePartQuatF)(q));
}

VEMA_IFC(void, ToLogVecQuatF)(VemaVec3F v, const VemaQuatF q) {
	VEMA_FN(ImPartQuatF)(v, q);
	VEMA_FN(NormalizeVec3F)(v);
	VEMA_FN(ScaleVec3F)(v, VEMA_FN(HalfAngQuatF)(q));
}

VEMA_IFC(void, FromLogVecQuatF)(VemaQuatF q, const VemaVec3F v) {
	VemaVec3F qv;
	float hang = VEMA_FN(FastMagVec3F)(v);
	float hcos = VEMA_FN(CosF)(hang);
#if 1
	VEMA_FN(ScaleSrcVec3F)(qv, v, VEMA_FN(SincF)(hang));
#else
	float hsin = VEMA_FN(SinF)(hang);
	VEMA_FN(NormalizeSrcVec3F)(qv, v);
	VEMA_FN(ScaleVec3F)(qv, hsin);
#endif
	VEMA_FN(SetPartsQuatF)(q, qv, hcos);
}


VEMA_IFC(void, CopyPlaneF)(VemaPlaneF dst, const VemaPlaneF src) {
	if (dst != src) {
		vemaMemCpy(dst, src, sizeof(VemaPlaneF));
	}
}

VEMA_IFC(void, PlaneFromPntNrmF)(VemaPlaneF pln, const VemaVec3F pos, const VemaVec3F nrm) {
	int i;
	for (i = 0; i < 3; ++i) {
		pln[i] = nrm[i];
	}
	pln[3] = VEMA_FN(DotVec3F)(pos, nrm);
}

VEMA_IFC(void, GetPlaneNormalF)(VemaVec3F nrm, const VemaPlaneF pln) {
	int i;
	for (i = 0; i < 3; ++i) {
		nrm[i] = pln[i];
	}
}

VEMA_IFC(float, SignedDistToPlane)(const VemaPlaneF pln, const VemaVec3F pos) {
	VemaVec3F nrm;
	VEMA_FN(GetPlaneNormalF)(nrm, pln);
	return VEMA_FN(DotVec3F)(pos, nrm) - pln[3];
}


VEMA_IFC(float, SqDistanceF)(const VemaVec3F p1, const VemaVec3F p2) {
	VemaVec3F v;
	VEMA_FN(SubVec3F)(v, p2, p1);
	return VEMA_FN(SqMagVec3F)(v);
}

VEMA_IFC(float, DistanceF)(const VemaVec3F p1, const VemaVec3F p2) {
	VemaVec3F v;
	VEMA_FN(SubVec3F)(v, p2, p1);
	return VEMA_FN(MagVec3F)(v);
}


VEMA_IFC(int, PointInRangeF)(const VemaVec3F pos, const VemaVec3F rmin, const VemaVec3F rmax) {
	int i;
	for (i = 0; i < 3; ++i) {
		if (pos[i] < rmin[i] || pos[i] > rmax[i]) return 0;
	}
	return 1;
}

VEMA_IFC(int, RangesOverlapF)(const VemaVec3F min1, const VemaVec3F max1, const VemaVec3F min2, const VemaVec3F max2) {
	int i;
	for (i = 0; i < 3; ++i) {
		if (min1[i] > max2[i] || max1[i] < min2[i]) return 0;
	}
	return 1;
}

VEMA_IFC(int, SegRangeOverlapF)(const VemaVec3F p0, const VemaVec3F p1, const VemaVec3F rmin, const VemaVec3F rmax) {
	VemaVec3F dir;
	float len;
	float tmin;
	float tmax;
	int i;
	VEMA_FN(SubVec3F)(dir, p1, p0);
	len = VEMA_FN(MagVec3F)(dir);
	if (len < 1.0e-7f) {
		return VEMA_FN(PointInRangeF)(p0, rmin, rmax);
	}
	tmin = 0.0f;
	tmax = len;
	for (i = 0; i < 3; ++i) {
		if (dir[i] != 0.0f) {
			float d = len / dir[i];
			float r1 = (rmin[i] - p0[i]) * d;
			float r2 = (rmax[i] - p0[i]) * d;
			float rmin = VEMA_FN(MinF)(r1, r2);
			float rmax = VEMA_FN(MaxF)(r1, r2);
			tmin = VEMA_FN(MaxF)(tmin, rmin);
			tmax = VEMA_FN(MinF)(tmax, rmax);
			if (tmin > tmax) {
				return 0;
			}
		} else {
			if (p0[i] < rmin[i] || p0[i] > rmax[i]) {
				return 0;
			}
		}
	}
	if (tmax > len) return 0;
	return 1;
}

VEMA_IFC(void, GetSphereCenterF)(VemaVec3F c, const VemaSphereF sph) {
	int i = 0;
	for (i = 0; i < 3; ++i) {
		c[i] = sph[i];
	}
}

VEMA_IFC(float, GetSphereRadiusF)(const VemaSphereF sph) {
	return sph[3];
}

VEMA_IFC(int, SpheresOverlapF)(const VemaSphereF sph1, const VemaSphereF sph2) {
	VemaVec3F c1;
	VemaVec3F c2;
	float rs = VEMA_FN(GetSphereRadiusF)(sph1) + VEMA_FN(GetSphereRadiusF)(sph2);
	VEMA_FN(GetSphereCenterF)(c1, sph1);
	VEMA_FN(GetSphereCenterF)(c2, sph2);
	return VEMA_FN(SqDistanceF)(c1, c2) <= VEMA_FN(SqF)(rs);
}

VEMA_IFC(int, OrientedBoxesOverlapF)(const VemaMtx4x4F m1, const VemaMtx4x4F m2) {
	VemaVec3F vx1;
	VemaVec3F vy1;
	VemaVec3F vz1;
	VemaVec3F t1;
	VemaVec3F r1;
	VemaVec3F vx2;
	VemaVec3F vy2;
	VemaVec3F vz2;
	VemaVec3F t2;
	VemaVec3F r2;
	VemaVec3F dv[3];
	VemaVec3F v;
	VemaVec3F tv;
	VemaVec3F cv;
	int i;
	float t;
	float x;
	float y;
	float z;

	VEMA_FN(GetAxisXMtx4x4F)(vx1, m1);
	VEMA_FN(GetAxisYMtx4x4F)(vy1, m1);
	VEMA_FN(GetAxisZMtx4x4F)(vz1, m1);
	VEMA_FN(GetTranslationMtx4x4F)(t1, m1);
	VEMA_FN(SetVec3F)(r1, VEMA_FN(MagVec3F)(vx1), VEMA_FN(MagVec3F)(vy1), VEMA_FN(MagVec3F)(vz1));
	VEMA_FN(ScaleVec3F)(r1, 0.5f);

	VEMA_FN(GetAxisXMtx4x4F)(vx2, m2);
	VEMA_FN(GetAxisYMtx4x4F)(vy2, m2);
	VEMA_FN(GetAxisZMtx4x4F)(vz2, m2);
	VEMA_FN(GetTranslationMtx4x4F)(t2, m2);
	VEMA_FN(SetVec3F)(r2, VEMA_FN(MagVec3F)(vx2), VEMA_FN(MagVec3F)(vy2), VEMA_FN(MagVec3F)(vz2));
	VEMA_FN(ScaleVec3F)(r2, 0.5f);

	VEMA_FN(SubVec3F)(v, t2, t1);
	if (VEMA_FN(SqMagVec3F)(v) <= VEMA_FN(SqF)(VEMA_FN(MinElemVec3F)(r1) + VEMA_FN(MinElemVec3F)(r2))) {
		return 1;
	}

	VEMA_FN(NormalizeVec3F)(vx1);
	VEMA_FN(NormalizeVec3F)(vy1);
	VEMA_FN(NormalizeVec3F)(vz1);
	VEMA_FN(NormalizeVec3F)(vx2);
	VEMA_FN(NormalizeVec3F)(vy2);
	VEMA_FN(NormalizeVec3F)(vz2);

	VEMA_FN(SetVec3F)(dv[0], VEMA_FN(DotVec3F)(vx1, vx2), VEMA_FN(DotVec3F)(vx1, vy2), VEMA_FN(DotVec3F)(vx1, vz2));
	VEMA_FN(AbsVec3F)(dv[0]);
	VEMA_FN(SetVec3F)(dv[1], VEMA_FN(DotVec3F)(vy1, vx2), VEMA_FN(DotVec3F)(vy1, vy2), VEMA_FN(DotVec3F)(vy1, vz2));
	VEMA_FN(AbsVec3F)(dv[1]);
	VEMA_FN(SetVec3F)(dv[2], VEMA_FN(DotVec3F)(vz1, vx2), VEMA_FN(DotVec3F)(vz1, vy2), VEMA_FN(DotVec3F)(vz1, vz2));
	VEMA_FN(AbsVec3F)(dv[2]);

	VEMA_FN(SetVec3F)(tv, VEMA_FN(DotVec3F)(v, vx1), VEMA_FN(DotVec3F)(v, vy1), VEMA_FN(DotVec3F)(v, vz1));
	VEMA_FN(AbsVec3F)(tv);
	VEMA_FN(SetVec3F)(cv, VEMA_FN(DotVec3F)(r2, dv[0]), VEMA_FN(DotVec3F)(r2, dv[1]), VEMA_FN(DotVec3F)(r2, dv[2]));
	VEMA_FN(AddVec3F)(cv, cv, r1);
	for (i = 0; i < 3; ++i) {
		if (tv[i] > cv[i]) return 0;
	}

	t = dv[0][1]; dv[0][1] = dv[1][0]; dv[1][0] = t;
	t = dv[0][2]; dv[0][2] = dv[2][0]; dv[2][0] = t;
	t = dv[1][2]; dv[1][2] = dv[2][1]; dv[2][1] = t;

	VEMA_FN(SetVec3F)(tv, VEMA_FN(DotVec3F)(v, vx2), VEMA_FN(DotVec3F)(v, vy2), VEMA_FN(DotVec3F)(v, vz2));
	VEMA_FN(AbsVec3F)(tv);
	VEMA_FN(SetVec3F)(cv, VEMA_FN(DotVec3F)(r1, dv[0]), VEMA_FN(DotVec3F)(r1, dv[1]), VEMA_FN(DotVec3F)(r1, dv[2]));
	VEMA_FN(AddVec3F)(cv, cv, r2);
	for (i = 0; i < 3; ++i) {
		if (tv[i] > cv[i]) return 0;
	}

	VEMA_FN(CrossVec3F)(dv[0], vx1, vx2);
	VEMA_FN(CrossVec3F)(dv[1], vx1, vy2);
	VEMA_FN(CrossVec3F)(dv[2], vx1, vz2);
	VEMA_FN(SetVec3F)(tv, VEMA_FN(DotVec3F)(dv[0], vx1), VEMA_FN(DotVec3F)(dv[0], vy1), VEMA_FN(DotVec3F)(dv[0], vz1));
	VEMA_FN(AbsVec3F)(tv);
	x = VEMA_FN(DotVec3F)(r1, tv);
	VEMA_FN(SetVec3F)(tv, VEMA_FN(DotVec3F)(dv[0], vx2), VEMA_FN(DotVec3F)(dv[0], vy2), VEMA_FN(DotVec3F)(dv[0], vz2));
	VEMA_FN(AbsVec3F)(tv);
	x += VEMA_FN(DotVec3F)(r2, tv);
	VEMA_FN(SetVec3F)(tv, VEMA_FN(DotVec3F)(dv[1], vx1), VEMA_FN(DotVec3F)(dv[1], vy1), VEMA_FN(DotVec3F)(dv[1], vz1));
	VEMA_FN(AbsVec3F)(tv);
	y = VEMA_FN(DotVec3F)(r1, tv);
	VEMA_FN(SetVec3F)(tv, VEMA_FN(DotVec3F)(dv[1], vx2), VEMA_FN(DotVec3F)(dv[1], vy2), VEMA_FN(DotVec3F)(dv[1], vz2));
	VEMA_FN(AbsVec3F)(tv);
	y += VEMA_FN(DotVec3F)(r2, tv);
	VEMA_FN(SetVec3F)(tv, VEMA_FN(DotVec3F)(dv[2], vx1), VEMA_FN(DotVec3F)(dv[2], vy1), VEMA_FN(DotVec3F)(dv[2], vz1));
	VEMA_FN(AbsVec3F)(tv);
	z = VEMA_FN(DotVec3F)(r1, tv);
	VEMA_FN(SetVec3F)(tv, VEMA_FN(DotVec3F)(dv[2], vx2), VEMA_FN(DotVec3F)(dv[2], vy2), VEMA_FN(DotVec3F)(dv[2], vz2));
	VEMA_FN(AbsVec3F)(tv);
	z += VEMA_FN(DotVec3F)(r2, tv);
	VEMA_FN(SetVec3F)(cv, x, y, z);
	VEMA_FN(SetVec3F)(tv, VEMA_FN(DotVec3F)(v, dv[0]), VEMA_FN(DotVec3F)(v, dv[1]), VEMA_FN(DotVec3F)(v, dv[2]));
	VEMA_FN(AbsVec3F)(tv);
	for (i = 0; i < 3; ++i) {
		if (tv[i] > cv[i]) return 0;
	}

	VEMA_FN(CrossVec3F)(dv[0], vy1, vx2);
	VEMA_FN(CrossVec3F)(dv[1], vy1, vy2);
	VEMA_FN(CrossVec3F)(dv[2], vy1, vz2);
	VEMA_FN(SetVec3F)(tv, VEMA_FN(DotVec3F)(dv[0], vx1), VEMA_FN(DotVec3F)(dv[0], vy1), VEMA_FN(DotVec3F)(dv[0], vz1));
	VEMA_FN(AbsVec3F)(tv);
	x = VEMA_FN(DotVec3F)(r1, tv);
	VEMA_FN(SetVec3F)(tv, VEMA_FN(DotVec3F)(dv[0], vx2), VEMA_FN(DotVec3F)(dv[0], vy2), VEMA_FN(DotVec3F)(dv[0], vz2));
	VEMA_FN(AbsVec3F)(tv);
	x += VEMA_FN(DotVec3F)(r2, tv);
	VEMA_FN(SetVec3F)(tv, VEMA_FN(DotVec3F)(dv[1], vx1), VEMA_FN(DotVec3F)(dv[1], vy1), VEMA_FN(DotVec3F)(dv[1], vz1));
	VEMA_FN(AbsVec3F)(tv);
	y = VEMA_FN(DotVec3F)(r1, tv);
	VEMA_FN(SetVec3F)(tv, VEMA_FN(DotVec3F)(dv[1], vx2), VEMA_FN(DotVec3F)(dv[1], vy2), VEMA_FN(DotVec3F)(dv[1], vz2));
	VEMA_FN(AbsVec3F)(tv);
	y += VEMA_FN(DotVec3F)(r2, tv);
	VEMA_FN(SetVec3F)(tv, VEMA_FN(DotVec3F)(dv[2], vx1), VEMA_FN(DotVec3F)(dv[2], vy1), VEMA_FN(DotVec3F)(dv[2], vz1));
	VEMA_FN(AbsVec3F)(tv);
	z = VEMA_FN(DotVec3F)(r1, tv);
	VEMA_FN(SetVec3F)(tv, VEMA_FN(DotVec3F)(dv[2], vx2), VEMA_FN(DotVec3F)(dv[2], vy2), VEMA_FN(DotVec3F)(dv[2], vz2));
	VEMA_FN(AbsVec3F)(tv);
	z += VEMA_FN(DotVec3F)(r2, tv);
	VEMA_FN(SetVec3F)(cv, x, y, z);
	VEMA_FN(SetVec3F)(tv, VEMA_FN(DotVec3F)(v, dv[0]), VEMA_FN(DotVec3F)(v, dv[1]), VEMA_FN(DotVec3F)(v, dv[2]));
	VEMA_FN(AbsVec3F)(tv);
	for (i = 0; i < 3; ++i) {
		if (tv[i] > cv[i]) return 0;
	}

	VEMA_FN(CrossVec3F)(dv[0], vz1, vx2);
	VEMA_FN(CrossVec3F)(dv[1], vz1, vy2);
	VEMA_FN(CrossVec3F)(dv[2], vz1, vz2);
	VEMA_FN(SetVec3F)(tv, VEMA_FN(DotVec3F)(dv[0], vx1), VEMA_FN(DotVec3F)(dv[0], vy1), VEMA_FN(DotVec3F)(dv[0], vz1));
	VEMA_FN(AbsVec3F)(tv);
	x = VEMA_FN(DotVec3F)(r1, tv);
	VEMA_FN(SetVec3F)(tv, VEMA_FN(DotVec3F)(dv[0], vx2), VEMA_FN(DotVec3F)(dv[0], vy2), VEMA_FN(DotVec3F)(dv[0], vz2));
	VEMA_FN(AbsVec3F)(tv);
	x += VEMA_FN(DotVec3F)(r2, tv);
	VEMA_FN(SetVec3F)(tv, VEMA_FN(DotVec3F)(dv[1], vx1), VEMA_FN(DotVec3F)(dv[1], vy1), VEMA_FN(DotVec3F)(dv[1], vz1));
	VEMA_FN(AbsVec3F)(tv);
	y = VEMA_FN(DotVec3F)(r1, tv);
	VEMA_FN(SetVec3F)(tv, VEMA_FN(DotVec3F)(dv[1], vx2), VEMA_FN(DotVec3F)(dv[1], vy2), VEMA_FN(DotVec3F)(dv[1], vz2));
	VEMA_FN(AbsVec3F)(tv);
	y += VEMA_FN(DotVec3F)(r2, tv);
	VEMA_FN(SetVec3F)(tv, VEMA_FN(DotVec3F)(dv[2], vx1), VEMA_FN(DotVec3F)(dv[2], vy1), VEMA_FN(DotVec3F)(dv[2], vz1));
	VEMA_FN(AbsVec3F)(tv);
	z = VEMA_FN(DotVec3F)(r1, tv);
	VEMA_FN(SetVec3F)(tv, VEMA_FN(DotVec3F)(dv[2], vx2), VEMA_FN(DotVec3F)(dv[2], vy2), VEMA_FN(DotVec3F)(dv[2], vz2));
	VEMA_FN(AbsVec3F)(tv);
	z += VEMA_FN(DotVec3F)(r2, tv);
	VEMA_FN(SetVec3F)(cv, x, y, z);
	VEMA_FN(SetVec3F)(tv, VEMA_FN(DotVec3F)(v, dv[0]), VEMA_FN(DotVec3F)(v, dv[1]), VEMA_FN(DotVec3F)(v, dv[2]));
	VEMA_FN(AbsVec3F)(tv);
	for (i = 0; i < 3; ++i) {
		if (tv[i] > cv[i]) return 0;
	}

	return 1;
}

VEMA_IFC(void, TriNormalCWF)(VemaVec3F nrm, const VemaVec3F v0, const VemaVec3F v1, const VemaVec3F v2) {
	VemaVec3F e0;
	VemaVec3F e1;
	VEMA_FN(SubVec3F)(e0, v0, v1);
	VEMA_FN(SubVec3F)(e1, v2, v1);
	VEMA_FN(CrossVec3F)(nrm, e0, e1);
	VEMA_FN(NormalizeVec3F)(nrm);
}

VEMA_IFC(void, TriNormalCCWF)(VemaVec3F nrm, const VemaVec3F v0, const VemaVec3F v1, const VemaVec3F v2) {
	VemaVec3F e0;
	VemaVec3F e1;
	VEMA_FN(SubVec3F)(e0, v1, v0);
	VEMA_FN(SubVec3F)(e1, v2, v0);
	VEMA_FN(CrossVec3F)(nrm, e0, e1);
	VEMA_FN(NormalizeVec3F)(nrm);
}

VEMA_IFC(int, SegQuadIntersectCCWF)(const VemaVec3F p0, const VemaVec3F p1, const VemaVec3F v0, const VemaVec3F v1, const VemaVec3F v2, const VemaVec3F v3, VemaVec3F hitPos, VemaVec3F hitNrm) {
	VemaVec3F vec0;
	VemaVec3F vec1;
	VemaVec3F vec2;
	VemaVec3F vec3;
	VemaVec3F edge0;
	VemaVec3F edge1;
	VemaVec3F edge2;
	VemaVec3F edge3;
	VemaVec3F nrm;
	VemaVec3F dir;
	float d0;
	float d1;
	float d;
	float t;
	VEMA_FN(ZeroVec3F)(hitPos);
	VEMA_FN(ZeroVec3F)(hitNrm);
	VEMA_FN(SubVec3F)(edge0, v1, v0);
	VEMA_FN(SubVec3F)(vec0, p0, v0);
	VEMA_FN(SubVec3F)(dir, v2, v0);
	VEMA_FN(CrossVec3F)(nrm, edge0, dir);
	VEMA_FN(NormalizeVec3F)(nrm);
	VEMA_FN(SubVec3F)(dir, p1, v0);
	d0 = VEMA_FN(DotVec3F)(vec0, nrm);
	d1 = VEMA_FN(DotVec3F)(dir, nrm);
	if (d0*d1 > 0.0f || (d0 == 0.0f && d1 == 0.0f)) return 0;
	VEMA_FN(SubVec3F)(dir, p1, p0);
	if (VEMA_FN(TripleVec3F)(edge0, dir, vec0) < 0.0f) return 0;
	VEMA_FN(SubVec3F)(edge1, v2, v1);
	VEMA_FN(SubVec3F)(vec1, p0, v1);
	if (VEMA_FN(TripleVec3F)(edge1, dir, vec1) < 0.0f) return 0;
	VEMA_FN(SubVec3F)(edge2, v3, v2);
	VEMA_FN(SubVec3F)(vec2, p0, v2);
	if (VEMA_FN(TripleVec3F)(edge2, dir, vec2) < 0.0f) return 0;
	VEMA_FN(SubVec3F)(edge3, v0, v3);
	VEMA_FN(SubVec3F)(vec3, p0, v3);
	if (VEMA_FN(TripleVec3F)(edge3, dir, vec3) < 0.0f) return 0;
	d = VEMA_FN(DotVec3F)(dir, nrm);
	if (d == 0.0f || d0 == 0.0f) {
		t = 0.0f;
	} else {
		t = -d0 / d;
	}
	if (t > 1.0f || t < 0.0f) return 0;
	VEMA_FN(ScaleSrcVec3F)(hitPos, dir, t);
	VEMA_FN(AddVec3F)(hitPos, hitPos, p0);
	VEMA_FN(CopyVec3F)(hitNrm, nrm);
	return 1;
}

VEMA_IFC(int, SegQuadIntersectCWF)(const VemaVec3F p0, const VemaVec3F p1, const VemaVec3F v0, const VemaVec3F v1, const VemaVec3F v2, const VemaVec3F v3, VemaVec3F hitPos, VemaVec3F hitNrm) {
	return VEMA_FN(SegQuadIntersectCCWF)(p0, p1, v3, v2, v1, v0, hitPos, hitNrm);
}

VEMA_IFC(int, SegTriIntersectCCWF)(const VemaVec3F p0, const VemaVec3F p1, const VemaVec3F v0, const VemaVec3F v1, const VemaVec3F v2, VemaVec3F hitPos, VemaVec3F hitNrm) {
	return VEMA_FN(SegQuadIntersectCCWF)(p0, p1, v0, v1, v2, v0, hitPos, hitNrm);
}

VEMA_IFC(int, SegTriIntersectCWF)(const VemaVec3F p0, const VemaVec3F p1, const VemaVec3F v0, const VemaVec3F v1, const VemaVec3F v2, VemaVec3F hitPos, VemaVec3F hitNrm) {
	return VEMA_FN(SegQuadIntersectCCWF)(p0, p1, v0, v2, v1, v0, hitPos, hitNrm);
}

VEMA_IFC(int, SegTriIntersectBarycentricCCWF)(const VemaVec3F p0, const VemaVec3F p1, const VemaVec3F v0, const VemaVec3F v1, const VemaVec3F v2, VemaVec4F uvwt) {
	VemaVec3F e1;
	VemaVec3F e2;
	VemaVec3F sv;
	VemaVec3F n;
	float d;
	int hit = 0;
	VEMA_FN(FillVec4F)(uvwt, -1.0f);
	VEMA_FN(SubVec3F)(e1, v1, v0);
	VEMA_FN(SubVec3F)(e2, v2, v0);
	VEMA_FN(SubVec3F)(sv, p0, p1);
	VEMA_FN(CrossVec3F)(n, e1, e2);
	d = VEMA_FN(DotVec3F)(sv, n);
	if (d > 0.0f) {
		VemaVec3F sv0;
		float t;
		VEMA_FN(SubVec3F)(sv0, p0, v0);
		t = VEMA_FN(DotVec3F)(sv0, n);
		if (t >= 0.0f && t <= d) {
			VemaVec3F e;
			float v;
			VEMA_FN(CrossVec3F)(e, sv, sv0);
			v = VEMA_FN(DotVec3F)(e, e2);
			if (v >= 0.0f && v <= d) {
				float w = -VEMA_FN(DotVec3F)(e, e1);
				if (w >= 0.0f && v + w <= d) {
					float s = 1.0f / d;
					VEMA_FN(SetVec4F)(uvwt, 1.0f, v, w, t);
					VEMA_FN(ScaleVec4F)(uvwt, s);
					uvwt[0] = 1.0f - uvwt[1] - uvwt[2];
					hit = 1;
				}
			}
		}
	}
	return hit;
}

VEMA_IFC(int, SegTriIntersectBarycentricCWF)(const VemaVec3F p0, const VemaVec3F p1, const VemaVec3F v0, const VemaVec3F v1, const VemaVec3F v2, VemaVec4F uvwt) {
	VemaVec3F e1;
	VemaVec3F e2;
	VemaVec3F sv;
	VemaVec3F n;
	float d;
	int hit = 0;
	VEMA_FN(FillVec4F)(uvwt, -1.0f);
	VEMA_FN(SubVec3F)(e1, v1, v0);
	VEMA_FN(SubVec3F)(e2, v2, v0);
	VEMA_FN(SubVec3F)(sv, p0, p1);
	VEMA_FN(CrossVec3F)(n, e2, e1);
	d = VEMA_FN(DotVec3F)(sv, n);
	if (d > 0.0f) {
		VemaVec3F sv0;
		float t;
		VEMA_FN(SubVec3F)(sv0, p0, v0);
		t = VEMA_FN(DotVec3F)(sv0, n);
		if (t >= 0.0f && t <= d) {
			VemaVec3F e;
			float w;
			VEMA_FN(CrossVec3F)(e, sv, sv0);
			w = VEMA_FN(DotVec3F)(e, e1);
			if (w >= 0.0f && w <= d) {
				float v = -VEMA_FN(DotVec3F)(e, e2);
				if (v >= 0.0f && v + w <= d) {
					float s = 1.0f / d;
					VEMA_FN(SetVec4F)(uvwt, 1.0f, v, w, t);
					VEMA_FN(ScaleVec4F)(uvwt, s);
					uvwt[0] = 1.0f - uvwt[1] - uvwt[2];
					hit = 1;
				}
			}
		}
	}
	return hit;
}


VEMA_IFC(void, ZeroSH3F)(VemaSH3F sh) {
	vemaMemSet(sh, 0, sizeof(VemaSH3F));
}

VEMA_IFC(void, CopySH3F)(VemaSH3F dst, const VemaSH3F src) {
	if (dst != src) {
		vemaMemCpy(dst, src, sizeof(VemaSH3F));
	}
}

VEMA_IFC(void, EvalSH3F)(VemaSH3F sh, const VemaVec3F v) {
	float x = v[0];
	float y = v[1];
	float z = v[2];
	float zz = z*z;
	float s0 = y;
	float c0 = x;
	float s1;
	float c1;
	float t = -0.31539156525252005f;
	sh[0] = 0.28209479177387814f;
	sh[2] = 0.4886025119029199f * z;
	sh[6] = 0.9461746957575601f * zz + t;
	t = -0.48860251190292f;
	sh[1] = t * s0;
	sh[3] = t * c0;
	t = -1.092548430592079f * z;
	sh[5] = t * s0;
	sh[7] = t * c0;
	s1 = x*s0 + y*c0;
	c1 = x*c0 - y*s0;
	t = 0.5462742152960395f;
	sh[4] = t * s1;
	sh[8] = t * c1;
}

VEMA_IFC(void, ScaleSH3F)(VemaSH3F sh, const float s) {
	int i;
	for (i = 0; i < 9; ++i) {
		sh[i] *= s;
	}
}

VEMA_IFC(uint32_t, EncodeRGBE)(const VemaVec3F rgb) {
	uint32_t rgbe = 0;
	float m = VEMA_FN(MaxElemVec3F)(rgb);
	if (m >= 1.0e-32f) {
		int ex;
		VemaVec3F t;
		float fr = VEMA_FN(FrExpF)(m, &ex);
		float s = VEMA_FN(Div0F)(fr * 256.0f, m);
		VEMA_FN(ScaleSrcVec3F)(t, rgb, s);
		rgbe = (uint8_t)t[0] & 0xFF;
		rgbe |= (uint32_t)((uint8_t)t[1] & 0xFF) << 8;
		rgbe |= (uint32_t)((uint8_t)t[2] & 0xFF) << 16;
		rgbe |= (uint32_t)((ex + 128) & 0xFF) << 24;
	}
	return rgbe;
}
