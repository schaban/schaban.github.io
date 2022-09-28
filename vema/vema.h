/* SPDX-License-Identifier: MIT */
/* SPDX-FileCopyrightText: 2020-2022 Sergey Chaban <sergey.chaban@gmail.com> */

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef VEMA_API_DECL
#	define VEMA_API_DECL
#endif

#ifndef VEMA_API_CALL
#	define VEMA_API_CALL
#endif

#define VEMA_FN(_name) vema##_name
#define VEMA_IFC(_ret, _name) VEMA_API_DECL _ret VEMA_API_CALL VEMA_FN(_name)

#define VEMA_PI_D 3.1415926535897931
#define VEMA_PI_F 3.14159274f

#define VEMA_AS_CONST_MTX(_m) *(const VemaMtx4x4F*)(_m)

typedef uint16_t VemaHalf;
typedef float VemaVec2F[2];
typedef float VemaVec3F[3];
typedef float VemaVec4F[4];
typedef float VemaMtx4x4F[4][4];
typedef float VemaMtx3x4F[3][4];
typedef float VemaQuatF[4];
typedef float VemaPlaneF[4];
typedef float VemaSphereF[4];
typedef float VemaSH3F[3 * 3];
typedef double VemaVec2D[2];
typedef double VemaVec3D[3];
typedef double VemaVec4D[4];

VEMA_IFC(uint32_t, Float32Bits)(const float x);
VEMA_IFC(float, Float32FromBits)(const uint32_t bits);
VEMA_IFC(VemaHalf, FloatToHalf)(const float x);
VEMA_IFC(float, HalfToFloat)(const VemaHalf h);
VEMA_IFC(void, HalfToFloatAry)(float* pDstF, const VemaHalf* pSrcH, const size_t n);

VEMA_IFC(void, FloatToU32Ary)(uint32_t* pDstU, const float* pSrcF, const size_t n);
VEMA_IFC(void, FloatToU32AryOffs)(uint32_t* pDstU, const size_t dstOffs, const float* pSrcF, const size_t srcOffs, const size_t n);
VEMA_IFC(void, FloatToU16Ary)(uint16_t* pDstU, const float* pSrcF, const size_t n);
VEMA_IFC(void, FloatToU16AryOffs)(uint16_t* pDstU, const size_t dstOffs, const float* pSrcF, const size_t srcOffs, const size_t n);

VEMA_IFC(int, AlmostEqualF)(const float x, const float y, const float tol);

VEMA_IFC(float, EvalPolynomial)(const float x, const float* pRevCoefs, const size_t n);

VEMA_IFC(float, FrExpF)(const float x, int* pExp);
VEMA_IFC(float, LdExpF)(const float x, const int e);

VEMA_IFC(float, InvalidF)(void);
VEMA_IFC(float, RadiansF)(const float deg);
VEMA_IFC(float, DegreesF)(const float rad);
VEMA_IFC(float, SqF)(const float x);
VEMA_IFC(float, CbF)(const float x);
VEMA_IFC(float, Div0F)(const float x, const float y);
VEMA_IFC(float, Rcp0F)(const float x);
VEMA_IFC(float, ModF)(const float x, const float y);
VEMA_IFC(float, AbsF)(const float x);
VEMA_IFC(float, MinF)(const float x, const float y);
VEMA_IFC(float, MaxF)(const float x, const float y);
VEMA_IFC(float, Min3F)(const float x, const float y, const float z);
VEMA_IFC(float, Max3F)(const float x, const float y, const float z);
VEMA_IFC(float, Min4F)(const float x, const float y, const float z, const float w);
VEMA_IFC(float, Max4F)(const float x, const float y, const float z, const float w);
VEMA_IFC(float, FloorF)(const float x);
VEMA_IFC(float, CeilF)(const float x);
VEMA_IFC(float, RoundF)(const float x);
VEMA_IFC(float, TruncF)(const float x);
VEMA_IFC(float, SqrtF)(const float x);
VEMA_IFC(float, SinF)(const float x);
VEMA_IFC(float, CosF)(const float x);
VEMA_IFC(float, TanF)(const float x);
VEMA_IFC(float, ArcSinF)(const float x);
VEMA_IFC(float, ArcCosF)(const float x);
VEMA_IFC(float, ArcTanF)(const float x);
VEMA_IFC(float, ArcTan2F)(const float y, const float x);
VEMA_IFC(float, SincF)(const float x);
VEMA_IFC(float, InvSincF)(const float x);
VEMA_IFC(float, HypotF)(const float x, const float y);
VEMA_IFC(float, Log10F)(const float x);
VEMA_IFC(float, LogF)(const float x);
VEMA_IFC(float, ExpF)(const float x);
VEMA_IFC(float, PowF)(const float x, const float y);
VEMA_IFC(float, IntPowF)(const float x, const int n);
VEMA_IFC(float, LerpF)(const float a, const float b, const float t);
VEMA_IFC(float, ClampF)(const float x, const float lo, const float hi);
VEMA_IFC(float, SaturateF)(const float x);
VEMA_IFC(float, FitF)(const float val, const float oldMin, const float oldMax, const float newMin, const float newMax);
VEMA_IFC(float, EaseCurveF)(const float p1, const float p2, const float t);

VEMA_IFC(double, RadiansD)(const double deg);
VEMA_IFC(double, DegreesD)(const double rad);
VEMA_IFC(double, SqD)(const double x);
VEMA_IFC(double, CbD)(const double x);
VEMA_IFC(double, Div0D)(const double x, const double y);
VEMA_IFC(double, Rcp0D)(const double x);
VEMA_IFC(double, ModD)(const double x, const double y);
VEMA_IFC(double, AbsD)(const double x);
VEMA_IFC(double, MinD)(const double x, const double y);
VEMA_IFC(double, MaxD)(const double x, const double y);
VEMA_IFC(double, Min3D)(const double x, const double y, const double z);
VEMA_IFC(double, Max3D)(const double x, const double y, const double z);
VEMA_IFC(double, Min4D)(const double x, const double y, const double z, const double w);
VEMA_IFC(double, Max4D)(const double x, const double y, const double z, const double w);
VEMA_IFC(double, FloorD)(const double x);
VEMA_IFC(double, CeilD)(const double x);
VEMA_IFC(double, RoundD)(const double x);
VEMA_IFC(double, SqrtD)(const double x);
VEMA_IFC(double, PowD)(const double x, const double y);
VEMA_IFC(double, SinD)(const double x);
VEMA_IFC(double, CosD)(const double x);
VEMA_IFC(double, ArcSinD)(const double x);
VEMA_IFC(double, ArcCosD)(const double x);

VEMA_IFC(void, TwoSumF)(VemaVec2F ts, const float x, const float y);

VEMA_IFC(void, SqrtVecF)(float* pDst, const int N);
VEMA_IFC(void, MulMtxF)(float* pDst, const float* pSrc1, const float* pSrc2, const int M, const int N, const int P);
VEMA_IFC(void, MulVecMtxF)(float* pDstVec, const float* pSrcVec, const float* pMtx, const int M, const int N);
VEMA_IFC(void, MulMtxVecF)(float* pDstVec, const float* pMtx, const float* pSrcVec, const int M, const int N);
VEMA_IFC(int, GJInvertMtxF)(float* pMtx, const int N, int* pWk /* [N*3] */);
VEMA_IFC(int, LUDecompMtxF)(float* pMtx, const int N, float* pWk /* [N] */, int* pIdx /* [N] */, int* pDetSgn);

VEMA_IFC(void, MulMtxD)(double* pDst, const double* pSrc1, const double* pSrc2, const int M, const int N, const int P);
VEMA_IFC(void, MulVecMtxD)(double* pDstVec, const double* pSrcVec, const double* pMtx, const int M, const int N);
VEMA_IFC(void, MulMtxVecD)(double* pDstVec, const double* pMtx, const double* pSrcVec, const int M, const int N);

VEMA_IFC(void, ZeroVec2F)(VemaVec2F v);
VEMA_IFC(void, CopyVec2F)(VemaVec2F dst, const VemaVec2F src);
VEMA_IFC(void, LoadVec2F)(VemaVec2F v, const void* pMem);
VEMA_IFC(void, StoreVec2F)(void* pMem, const VemaVec2F v);
VEMA_IFC(void, LoadAtIdxVec2F)(VemaVec2F v, const void* pMem, const int32_t idx);
VEMA_IFC(void, StoreAtIdxVec2F)(void* pMem, const int32_t idx, const VemaVec2F v);
VEMA_IFC(void, LoadAtOffsVec2F)(VemaVec2F v, const void* pMem, const size_t offs);
VEMA_IFC(void, StoreAtOffsVec2F)(void* pMem, const size_t offs, const VemaVec2F v);

VEMA_IFC(void, ZeroVec3F)(VemaVec3F v);
VEMA_IFC(void, FillVec3F)(VemaVec3F v, const float s);
VEMA_IFC(void, CopyVec3F)(VemaVec3F dst, const VemaVec3F src);
VEMA_IFC(void, SetVec3F)(VemaVec3F v, const float x, const float y, const float z);
VEMA_IFC(void, LoadVec3F)(VemaVec3F v, const void* pMem);
VEMA_IFC(void, StoreVec3F)(void* pMem, const VemaVec3F v);
VEMA_IFC(void, LoadAtIdxVec3F)(VemaVec3F v, const void* pMem, const int32_t idx);
VEMA_IFC(void, StoreAtIdxVec3F)(void* pMem, const int32_t idx, const VemaVec3F v);
VEMA_IFC(void, LoadAtOffsVec3F)(VemaVec3F v, const void* pMem, const size_t offs);
VEMA_IFC(void, StoreAtOffsVec3F)(void* pMem, const size_t offs, const VemaVec3F v);
VEMA_IFC(void, AddVec3F)(VemaVec3F v0, const VemaVec3F v1, const VemaVec3F v2);
VEMA_IFC(void, SubVec3F)(VemaVec3F v0, const VemaVec3F v1, const VemaVec3F v2);
VEMA_IFC(void, MulVec3F)(VemaVec3F v0, const VemaVec3F v1, const VemaVec3F v2);
VEMA_IFC(void, CrossVec3F)(VemaVec3F v0, const VemaVec3F v1, const VemaVec3F v2);
VEMA_IFC(void, ScaleVec3F)(VemaVec3F v, const float s);
VEMA_IFC(void, ScaleSrcVec3F)(VemaVec3F v, const VemaVec3F vsrc, const float s);
VEMA_IFC(float, DotVec3F)(const VemaVec3F v1, const VemaVec3F v2);
VEMA_IFC(float, TripleVec3F)(const VemaVec3F v0, const VemaVec3F v1, const VemaVec3F v2);
VEMA_IFC(float, MinElemVec3F)(const VemaVec3F v);
VEMA_IFC(float, MaxElemVec3F)(const VemaVec3F v);
VEMA_IFC(float, SqMagVec3F)(const VemaVec3F v);
VEMA_IFC(float, FastMagVec3F)(const VemaVec3F v);
VEMA_IFC(float, MagVec3F)(const VemaVec3F v);
VEMA_IFC(void, NormalizeVec3F)(VemaVec3F v);
VEMA_IFC(void, NormalizeSrcVec3F)(VemaVec3F v, const VemaVec3F src);
VEMA_IFC(void, NegVec3F)(VemaVec3F v);
VEMA_IFC(void, AbsVec3F)(VemaVec3F v);
VEMA_IFC(void, Rcp0Vec3F)(VemaVec3F v);
VEMA_IFC(void, LerpVec3F)(VemaVec3F v0, const VemaVec3F v1, const VemaVec3F v2, const float bias);
VEMA_IFC(void, ClampVec3F)(VemaVec3F v, const VemaVec3F vmin, const VemaVec3F vmax);
VEMA_IFC(void, SaturateVec3F)(VemaVec3F v);
VEMA_IFC(void, MinVec3F)(VemaVec3F v0, const VemaVec3F v1, const VemaVec3F v2);
VEMA_IFC(void, MaxVec3F)(VemaVec3F v0, const VemaVec3F v1, const VemaVec3F v2);
VEMA_IFC(void, CombineVec3F)(VemaVec3F v0, const VemaVec3F v1, const float s1, const VemaVec3F v2, const float s2);
VEMA_IFC(void, RotXYVec3F)(VemaVec3F v, const float rx, const float ry);
VEMA_IFC(int, AlmostEqualVec3F)(const VemaVec3F v1, const VemaVec3F v2, const float tol);

VEMA_IFC(void, ZeroVec4F)(VemaVec4F v);
VEMA_IFC(void, FillVec4F)(VemaVec4F v, const float s);
VEMA_IFC(void, CopyVec4F)(VemaVec4F dst, const VemaVec4F src);
VEMA_IFC(void, SetVec4F)(VemaVec4F v, const float x, const float y, const float z, const float w);
VEMA_IFC(void, LoadVec4F)(VemaVec4F v, const void* pMem);
VEMA_IFC(void, StoreVec4F)(void* pMem, const VemaVec4F v);
VEMA_IFC(void, LoadAtIdxVec4F)(VemaVec4F v, const void* pMem, const int32_t idx);
VEMA_IFC(void, StoreAtIdxVec4F)(void* pMem, const int32_t idx, const VemaVec4F v);
VEMA_IFC(void, LoadAtOffsVec4F)(VemaVec4F v, const void* pMem, const size_t offs);
VEMA_IFC(void, StoreAtOffsVec4F)(void* pMem, const size_t offs, const VemaVec4F v);
VEMA_IFC(void, ScaleVec4F)(VemaVec4F v, const float s);
VEMA_IFC(void, SaturateVec4F)(VemaVec4F v);

VEMA_IFC(void, ZeroVec3D)(VemaVec3D v);
VEMA_IFC(void, CopyVec3D)(VemaVec3D dst, const VemaVec3D src);
VEMA_IFC(void, SetVec3D)(VemaVec3D v, const double x, const double y, const double z);
VEMA_IFC(void, AddVec3D)(VemaVec3D v0, const VemaVec3D v1, const VemaVec3D v2);
VEMA_IFC(void, SubVec3D)(VemaVec3D v0, const VemaVec3D v1, const VemaVec3D v2);
VEMA_IFC(void, MulVec3D)(VemaVec3D v0, const VemaVec3D v1, const VemaVec3D v2);
VEMA_IFC(void, CrossVec3D)(VemaVec3D v0, const VemaVec3D v1, const VemaVec3D v2);
VEMA_IFC(void, ScaleVec3D)(VemaVec3D v, const double s);
VEMA_IFC(void, ScaleSrcVec3D)(VemaVec3D v, const VemaVec3D vsrc, const double s);
VEMA_IFC(double, DotVec3D)(const VemaVec3D v1, const VemaVec3D v2);
VEMA_IFC(double, TripleVec3D)(const VemaVec3D v0, const VemaVec3D v1, const VemaVec3D v2);
VEMA_IFC(double, SqMagVec3D)(const VemaVec3D v);
VEMA_IFC(double, FastMagVec3D)(const VemaVec3D v);
VEMA_IFC(double, MagVec3D)(const VemaVec3D v);
VEMA_IFC(void, NormalizeVec3D)(VemaVec3D v);
VEMA_IFC(void, NormalizeSrcVec3D)(VemaVec3D v, const VemaVec3D src);

VEMA_IFC(void, ZeroMtx4x4F)(VemaMtx4x4F m);
VEMA_IFC(void, IdentityMtx4x4F)(VemaMtx4x4F m);
VEMA_IFC(void, CopyMtx4x4F)(VemaMtx4x4F dst, const VemaMtx4x4F src);
VEMA_IFC(void, LoadMtx4x4F)(VemaMtx4x4F m, const void* pMem);
VEMA_IFC(void, StoreMtx4x4F)(void* pMem, const VemaMtx4x4F m);
VEMA_IFC(void, LoadAtIdxMtx4x4F)(VemaMtx4x4F m, const void* pMem, const int32_t idx);
VEMA_IFC(void, StoreAtIdxMtx4x4F)(void* pMem, const int32_t idx, const VemaMtx4x4F m);
VEMA_IFC(void, LoadAtOffsMtx4x4F)(VemaMtx4x4F m, const void* pMem, const size_t offs);
VEMA_IFC(void, StoreAtOffsMtx4x4F)(void* pMem, const size_t offs, const VemaMtx4x4F m);
VEMA_IFC(void, MulMtx4x4F)(VemaMtx4x4F m0, const VemaMtx4x4F m1, const VemaMtx4x4F m2);
VEMA_IFC(void, MulAryMtx4x4F)(VemaMtx4x4F* pDst, const VemaMtx4x4F* pSrc1, const VemaMtx4x4F* pSrc2, const size_t n);
VEMA_IFC(void, TransposeMtx4x4F)(VemaMtx4x4F m);
VEMA_IFC(void, TransposeSrcMtx4x4F)(VemaMtx4x4F m, const VemaMtx4x4F src);
VEMA_IFC(void, TransposeAxesMtx4x4F)(VemaMtx4x4F m);
VEMA_IFC(void, InvertMtx4x4F)(VemaMtx4x4F m);
VEMA_IFC(void, InvertSrcMtx4x4F)(VemaMtx4x4F m, const VemaMtx4x4F src);
VEMA_IFC(void, NrmAxisRotMtx4x4F)(VemaMtx4x4F m, const VemaVec3F v, const float ang);
VEMA_IFC(void, AxisRotMtx4x4F)(VemaMtx4x4F m, const VemaVec3F axis, const float ang);
VEMA_IFC(void, RotXMtx4x4F)(VemaMtx4x4F m, const float ang);
VEMA_IFC(void, RotYMtx4x4F)(VemaMtx4x4F m, const float ang);
VEMA_IFC(void, RotZMtx4x4F)(VemaMtx4x4F m, const float ang);
VEMA_IFC(void, AxisXMtx4x4F)(VemaMtx4x4F m, const VemaVec3F v);
VEMA_IFC(void, AxisXValsMtx4x4F)(VemaMtx4x4F m, const float x, const float y, const float z);
VEMA_IFC(void, AxisYMtx4x4F)(VemaMtx4x4F m, const VemaVec3F v);
VEMA_IFC(void, AxisYValsMtx4x4F)(VemaMtx4x4F m, const float x, const float y, const float z);
VEMA_IFC(void, AxisZMtx4x4F)(VemaMtx4x4F m, const VemaVec3F v);
VEMA_IFC(void, AxisZValsMtx4x4F)(VemaMtx4x4F m, const float x, const float y, const float z);
VEMA_IFC(void, QuatAxesMtx4x4F)(VemaMtx4x4F m, const VemaQuatF q);
VEMA_IFC(void, TranslationMtx4x4F)(VemaMtx4x4F m, const VemaVec3F t);
VEMA_IFC(void, TranslationValsMtx4x4F)(VemaMtx4x4F m, const float tx, const float ty, const float tz);
VEMA_IFC(void, ZeroTranslationMtx4x4F)(VemaMtx4x4F m);
VEMA_IFC(void, GetAxisXMtx4x4F)(VemaVec3F v, const VemaMtx4x4F m);
VEMA_IFC(void, GetAxisYMtx4x4F)(VemaVec3F v, const VemaMtx4x4F m);
VEMA_IFC(void, GetAxisZMtx4x4F)(VemaVec3F v, const VemaMtx4x4F m);
VEMA_IFC(void, GetTranslationMtx4x4F)(VemaVec3F v, const VemaMtx4x4F m);
VEMA_IFC(void, ScalingMtx4x4F)(VemaMtx4x4F m, const float sx, const float sy, const float sz);
VEMA_IFC(void, UniformScalingMtx4x4F)(VemaMtx4x4F m, const float s);
VEMA_IFC(void, ViewMtx4x4F)(VemaMtx4x4F m, const VemaVec3F pos, const VemaVec3F tgt, const VemaVec3F up, VemaMtx4x4F* pInv);
VEMA_IFC(void, PerspectiveMtx4x4F)(VemaMtx4x4F m, const float fovy, const float aspect, const float znear, const float zfar);
VEMA_IFC(void, CalcVecMtx4x4F)(VemaVec3F dst, const VemaVec3F src, const VemaMtx4x4F m);
VEMA_IFC(void, CalcPntMtx4x4F)(VemaVec3F dst, const VemaVec3F src, const VemaMtx4x4F m);
VEMA_IFC(void, FromQuatMtx4x4F)(VemaMtx4x4F m, const VemaQuatF q);

VEMA_IFC(void, ZeroMtx3x4F)(VemaMtx3x4F m);
VEMA_IFC(void, IdentityMtx3x4F)(VemaMtx3x4F m);
VEMA_IFC(void, CopyMtx3x4F)(VemaMtx3x4F dst, const VemaMtx3x4F src);
VEMA_IFC(void, ConvertMtx4x4FtoMtx3x4F)(VemaMtx3x4F dst, const VemaMtx4x4F src);
VEMA_IFC(void, ConcatMtx3x4F)(VemaMtx3x4F m0, const VemaMtx3x4F m1, const VemaMtx3x4F m2);

VEMA_IFC(void, IdentityQuatF)(VemaQuatF q);
VEMA_IFC(void, CopyQuatF)(VemaQuatF dst, const VemaQuatF src);
VEMA_IFC(void, SetQuatF)(VemaQuatF q, const float x, const float y, const float z, const float w);
VEMA_IFC(void, SetPartsQuatF)(VemaQuatF q, const VemaVec3F im, const float re);
VEMA_IFC(void, AxisRotQuatF)(VemaQuatF q, const VemaVec3F axis, const float ang);
VEMA_IFC(void, RotXQuatF)(VemaQuatF q, const float ang);
VEMA_IFC(void, RotYQuatF)(VemaQuatF q, const float ang);
VEMA_IFC(void, RotZQuatF)(VemaQuatF q, const float ang);
VEMA_IFC(void, GetAxisXQuatF)(VemaVec3F v, const VemaQuatF q);
VEMA_IFC(void, GetAxisYQuatF)(VemaVec3F v, const VemaQuatF q);
VEMA_IFC(void, GetAxisZQuatF)(VemaVec3F v, const VemaQuatF q);
VEMA_IFC(void, MulQuatF)(VemaQuatF q0, const VemaQuatF q1, const VemaQuatF q2);
VEMA_IFC(void, SlerpQuatF)(VemaQuatF q, const VemaQuatF q1, const VemaQuatF q2, const float bias);
VEMA_IFC(void, ScaleQuatF)(VemaQuatF q, const float s);
VEMA_IFC(void, NormalizeQuatF)(VemaQuatF q);
VEMA_IFC(void, NegQuatF)(VemaQuatF q);
VEMA_IFC(void, ConjugateQuatF)(VemaQuatF q);
VEMA_IFC(float, DotQuatF)(const VemaQuatF q1, const VemaQuatF q2);
VEMA_IFC(float, RePartQuatF)(const VemaQuatF q);
VEMA_IFC(void, ImPartQuatF)(VemaVec3F v, const VemaQuatF q);
VEMA_IFC(void, FromMtx4x4QuatF)(VemaQuatF q, const VemaMtx4x4F m);
VEMA_IFC(void, ApplyQuatF)(VemaVec3F dst, const VemaVec3F src, const VemaQuatF q);
VEMA_IFC(float, FastMagQuatF)(const VemaQuatF q);
VEMA_IFC(float, HalfAngQuatF)(const VemaQuatF q);
VEMA_IFC(void, ToLogVecQuatF)(VemaVec3F v, const VemaQuatF q);
VEMA_IFC(void, FromLogVecQuatF)(VemaQuatF q, const VemaVec3F v);

VEMA_IFC(void, CopyPlaneF)(VemaPlaneF dst, const VemaPlaneF src);
VEMA_IFC(void, PlaneFromPntNrmF)(VemaPlaneF pln, const VemaVec3F pos, const VemaVec3F nrm);
VEMA_IFC(void, GetPlaneNormalF)(VemaVec3F nrm, const VemaPlaneF pln);
VEMA_IFC(float, SignedDistToPlane)(const VemaPlaneF pln, const VemaVec3F pos);

VEMA_IFC(float, SqDistanceF)(const VemaVec3F p1, const VemaVec3F p2);
VEMA_IFC(float, DistanceF)(const VemaVec3F p1, const VemaVec3F p2);

VEMA_IFC(int, PointInRangeF)(const VemaVec3F pos, const VemaVec3F rmin, const VemaVec3F rmax);
VEMA_IFC(int, RangesOverlapF)(const VemaVec3F min1, const VemaVec3F max1, const VemaVec3F min2, const VemaVec3F max2);
VEMA_IFC(int, SegRangeOverlapF)(const VemaVec3F p0, const VemaVec3F p1, const VemaVec3F rmin, const VemaVec3F rmax);

VEMA_IFC(void, GetSphereCenterF)(VemaVec3F c, const VemaSphereF sph);
VEMA_IFC(float, GetSphereRadiusF)(const VemaSphereF sph);
VEMA_IFC(int, SpheresOverlapF)(const VemaSphereF sph1, const VemaSphereF sph2);

VEMA_IFC(int, OrientedBoxesOverlapF)(const VemaMtx4x4F m1, const VemaMtx4x4F m2);

VEMA_IFC(void, TriNormalCWF)(VemaVec3F nrm, const VemaVec3F v0, const VemaVec3F v1, const VemaVec3F v2);
VEMA_IFC(void, TriNormalCCWF)(VemaVec3F nrm, const VemaVec3F v0, const VemaVec3F v1, const VemaVec3F v2);

VEMA_IFC(int, SegQuadIntersectCCWF)(const VemaVec3F p0, const VemaVec3F p1, const VemaVec3F v0, const VemaVec3F v1, const VemaVec3F v2, const VemaVec3F v3, VemaVec3F hitPos, VemaVec3F hitNrm);
VEMA_IFC(int, SegQuadIntersectCWF)(const VemaVec3F p0, const VemaVec3F p1, const VemaVec3F v0, const VemaVec3F v1, const VemaVec3F v2, const VemaVec3F v3, VemaVec3F hitPos, VemaVec3F hitNrm);
VEMA_IFC(int, SegTriIntersectCCWF)(const VemaVec3F p0, const VemaVec3F p1, const VemaVec3F v0, const VemaVec3F v1, const VemaVec3F v2, VemaVec3F hitPos, VemaVec3F hitNrm);
VEMA_IFC(int, SegTriIntersectCWF)(const VemaVec3F p0, const VemaVec3F p1, const VemaVec3F v0, const VemaVec3F v1, const VemaVec3F v2, VemaVec3F hitPos, VemaVec3F hitNrm);
VEMA_IFC(int, SegTriIntersectBarycentricCCWF)(const VemaVec3F p0, const VemaVec3F p1, const VemaVec3F v0, const VemaVec3F v1, const VemaVec3F v2, VemaVec4F uvwt);
VEMA_IFC(int, SegTriIntersectBarycentricCWF)(const VemaVec3F p0, const VemaVec3F p1, const VemaVec3F v0, const VemaVec3F v1, const VemaVec3F v2, VemaVec4F uvwt);

VEMA_IFC(void, ZeroSH3F)(VemaSH3F sh);
VEMA_IFC(void, CopySH3F)(VemaSH3F dst, const VemaSH3F src);
VEMA_IFC(void, EvalSH3F)(VemaSH3F sh, const VemaVec3F v);
VEMA_IFC(void, ScaleSH3F)(VemaSH3F sh, const float s);

VEMA_IFC(uint32_t, EncodeRGBE)(const VemaVec3F rgb);

#ifdef __cplusplus
}
#endif

