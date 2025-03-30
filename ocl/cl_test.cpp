#include "crosscore.hpp"
#include "oglsys.hpp"
#include "cl_util.hpp"

namespace {

// FIXME: F3 -> F4
static HostMem<xt_float3> f32x3_in_mem(OGLSys::CL::Context ctx, const size_t n) {
	return HostMem<xt_float3>::create_in(ctx, n);
}

static HostMem<xt_float3> f32x3_out_mem(OGLSys::CL::Context ctx, const size_t n) {
	return HostMem<xt_float3>::create_out(ctx, n);
}


static const char* s_seg_tri_kern = "\
	float3 vcross(float3 v1, float3 v2) { return v1.yzx*v2.zxy - v1.zxy*v2.yzx; }\n\
	__kernel void seg_tri_kern(__global float4* pRes, __global float* pPos, __global float* pVtx) {\n\
		int idx = get_global_id(0);\n\
		int iseg = idx * 6;\n\
		float3 p0 = (float3){ pPos[iseg + 0], pPos[iseg + 1], pPos[iseg + 2] };\n\
		float3 p1 = (float3){ pPos[iseg + 3], pPos[iseg + 4], pPos[iseg + 5] };\n\
		int itri = idx * 9;\n\
		float3 v0 = (float3){ pVtx[itri + 0], pVtx[itri + 1], pVtx[itri + 2] };\n\
		float3 v1 = (float3){ pVtx[itri + 3], pVtx[itri + 4], pVtx[itri + 5] };\n\
		float3 v2 = (float3){ pVtx[itri + 6], pVtx[itri + 7], pVtx[itri + 8] };\n\
		float4 uvwt = (float4){ -1.0f, -1.0f, -1.0f, -1.0f };\n\
		float3 e1 = v1 - v0;\n\
		float3 e2 = v2 - v0;\n\
		float3 sv = p0 - p1;\n\
		float3 n = vcross(e2, e1);\n\
		float d = dot(sv, n);\n\
		if (d > 0.0f) {\n\
			float3 sv0 = p0 - v0;\n\
			float t = dot(sv0, n);\n\
			if (t >= 0.0f && t <= d) {\n\
				float3 e = vcross(sv, sv0);\n\
				float w = dot(e, e1);\n\
				if (w >= 0.0f && w <= d) {\n\
					float v = -dot(e, e2);\n\
					if (v >= 0.0f && v + w <= d) {\n\
						float s = native_recip(d);\n\
						uvwt.yzw = (float3)(v, w, t) * s;\n\
						uvwt.x = 1.0f - uvwt.y - uvwt.z;\n\
					}\n\
				}\n\
			}\n\
		}\n\
		pRes[idx] = uvwt;\n\
	}\n\
";

static const char* s_f3_add_kern = "\
	__kernel void f3_add_kern(__global float3* pRes, float3 vec1, float3 vec2) {\n\
		*pRes = vec1 + vec2;\n\
		pRes[1] = (float3)(-0.1f, -0.2f, -0.3f);\n\
	}\n\
";

static const char* s_dot_kern = "\
	__kernel void dot_kern(__global float* pRes, __global float* pVec1, __global float* pVec2, int numElems) {\n\
		int idx = get_global_id(0);\n\
		int isrc = idx * numElems;\n\
		float d = 0.0f;\n\
		for (int i = 0; i < numElems; ++i) {\n\
			d += pVec1[isrc + i] * pVec2[isrc + i];\n\
		}\n\
		pRes[idx] = d;\n\
	}\n\
";

static const char* s_dot_kern_half = "\
	#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n\
	__kernel void dot_kern_half(__global half* pRes, __global half* pVec1, __global half* pVec2, int numElems) {\n\
		int idx = get_global_id(0);\n\
		int isrc = idx * numElems;\n\
		half d = 0.0h;\n\
		for (int i = 0; i < numElems; ++i) {\n\
			d += pVec1[isrc + i] * pVec2[isrc + i];\n\
		}\n\
		pRes[idx] = d;\n\
	}\n\
";

static const char* s_dot_kern_dbl = "\
	#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n\
	__kernel void dot_kern_dbl(__global double* pRes, __global double* pVec1, __global double* pVec2, int numElems) {\n\
		int idx = get_global_id(0);\n\
		int isrc = idx * numElems;\n\
		double d = 0.0;\n\
		for (int i = 0; i < numElems; ++i) {\n\
			d += pVec1[isrc + i] * pVec2[isrc + i];\n\
		}\n\
		pRes[idx] = d;\n\
	}\n\
";

XD_NOINLINE static void cpu_dot(float* pRes, const float* pVec1, const float* pVec2, const int numElems, const int numVecs) {
	for (int i = 0; i < numVecs; ++i) {
		int offs = i * numElems;
		pRes[i] = 0.0f;
		for (int j = 0; j < numElems; ++j) {
			pRes[i] += pVec1[offs + j] * pVec2[offs + j];
		}
	}
}


XD_NOINLINE static void seg_tri_cpu_loop(xt_float4* pRes, cxVec* pSegs, cxVec* pTris, const int num) {
	for (int i = 0; i < num; ++i) {
		int itri = i * 3;
		cxVec v0 = pTris[itri + 0];
		cxVec v1 = pTris[itri + 1];
		cxVec v2 = pTris[itri + 2];
		int iseg = i * 2;
		cxVec p0 = pSegs[iseg + 0];
		cxVec p1 = pSegs[iseg + 1];
		pRes[i] = nxGeom::seg_tri_intersect_bc_cw(p0, p1, v0, v1, v2);
	}
}

static void seg_tri_test(OGLSys::CL::Context ctx, OGLSys::CL::Queue que, const int num) {
	if (!ctx) return;
	if (!que) return;
	const char* pOpts = nullptr;
	OGLSys::CL::Kernel kern = OGLSys::CL::create_kernel_from_src(ctx, s_seg_tri_kern, "seg_tri_kern", pOpts);
	if (!kern) return;
	int numTriVecs = num * 3;
	cxVec* pTris = nxCore::tMem<cxVec>::alloc(numTriVecs, "CPU:tris");
	int numSegVecs = num * 2;
	cxVec* pSegs = nxCore::tMem<cxVec>::alloc(numSegVecs, "CPU:segs");
	xt_float4* pRes = nxCore::tMem<xt_float4>::alloc(num, "CPU:UVWT");
	cxVec xpos0(-1.1f, -2.2f, -3.3f);
	cxVec xrot0(-10.0f, -20.0f, -30.0f);
	cxVec xpos1(1.1f, 2.2f, 3.3f);
	cxVec xrot1(10.0f, 20.0f, 30.0f);
	cxVec uv0(0.1f, 0.2f, 0.0f);
	cxVec uv1(0.3f, 0.5f, 0.0f);
	cxVec rel0(0.75f, 0.25f, 0.0f);
	cxVec rel1(0.1f, 0.3f, 0.0f);
	float xt = 0.0f;
	float xadd = nxCalc::rcp0(float(num - 1));
	for (int i = 0; i < num; ++i) {
		cxVec xpos = nxVec::lerp(xpos0, xpos1, xt);
		cxVec xrot = nxVec::lerp(xrot0, xrot1, xt);
		cxVec uv = nxVec::lerp(uv0, uv1, xt);
		cxVec rel = nxVec::lerp(rel0, rel1, xt);
		cxMtx xform = nxMtx::mk_rot_degrees(xrot) * nxMtx::mk_pos(xpos);
		cxVec v0(0.0f, 0.0f, 0.0f);
		cxVec v1(1.0f, 0.0f, 0.0f);
		cxVec v2(0.0f, 0.0f, 1.0f);
		cxVec tn(0.0f, 1.0f, 0.0f);
		v0 = xform.calc_pnt(v0);
		v1 = xform.calc_pnt(v1);
		v2 = xform.calc_pnt(v2);
		tn = xform.calc_vec(tn).get_normalized();
		int itri = i * 3;
		pTris[itri + 0] = v0;
		pTris[itri + 1] = v1;
		pTris[itri + 2] = v2;
		float tu = uv.x;
		float tv = uv.y;
		float tw = 1.0f - tu - tv;
		cxVec tpos = v0*tu + v1*tv + v2*tw;
		cxVec p0 = tpos + (tn * rel.x);
		cxVec p1 = tpos - (tn * rel.y);
		int iseg = i * 2;
		pSegs[iseg + 0] = p0;
		pSegs[iseg + 1] = p1;
		xt += xadd;
	}

	bool fastIn = true;
	bool fastOut = false;

	size_t numTriItems = num * 3 * 3;
	auto triIn = f32_in_mem(ctx, numTriItems, fastIn);
	if (triIn.mpMem) {
		::memcpy(triIn.mpMem, pTris, numTriItems * sizeof(float));
	}
	triIn.update(que);

	size_t numSegItems = num * 3 * 2;
	auto segIn = f32_in_mem(ctx, numSegItems, fastIn);
	if (segIn.mpMem) {
		::memcpy(segIn.mpMem, pSegs, numSegItems * sizeof(float));
	}
	segIn.update(que);

	auto resOut = f32x4_out_mem(ctx, num, fastOut);

	nxCore::dbg_msg(
		"[CL seg-tri] in seg: %s, in tri: %s, out: %s\n",
		segIn.mIsFast ? "fast" : "std",
		triIn.mIsFast ? "fast" : "std",
		resOut.mIsFast ? "fast" : "std");

	OGLSys::CL::set_kernel_buffer_arg(kern, 0, resOut.mBuf);
	OGLSys::CL::set_kernel_buffer_arg(kern, 1, segIn.mBuf);
	OGLSys::CL::set_kernel_buffer_arg(kern, 2, triIn.mBuf);

	/* warm up */
	OGLSys::CL::exec_kernel(que, kern, 1);
	resOut.force_update(que);

	OGLSys::CL::Event evt;
	OGLSys::CL::exec_kernel(que, kern, num, &evt);
	OGLSys::CL::flush_queue(que);

	double t0 = nxSys::time_micros();
	seg_tri_cpu_loop(pRes, pSegs, pTris, num);
	double dtCPU = nxSys::time_micros() - t0;

	int nwait = 0;
	while (true) {
		if (OGLSys::CL::event_ck_complete(evt)) {
			break;
		}
		++nwait;
	}
	resOut.update(que);

	double dt = nxSys::time_micros() - t0;
	nxCore::dbg_msg("[CL seg-tri] dt: %.3f micros (CPU: %.3f)\n", dt, dtCPU);

	float maxDiff = 0.0f;
	for (int i = 0; i < num; ++i) {
		for (int j = 0; j < 4; ++j) {
			float valCPU = pRes[i].get_at(j);
			float valAcc = resOut.mpMem[i].get_at(j);
			bool eq = nxCore::f32_almost_eq(valCPU, valAcc);
			if (!eq) {
				nxCore::dbg_msg("[CL seg-tri] CPU/Acc mismatch @ %d\n", i);
			}
			maxDiff = nxCalc::max(maxDiff, ::mth_fabsf(valCPU - valAcc));
		}
	}
	nxCore::dbg_msg("[CL seg-tri] max diff = %.12f)\n", maxDiff);

	resOut.reset();
	segIn.reset();
	triIn.reset();
	OGLSys::CL::release_kernel(kern);
	nxCore::tMem<xt_float4>::free(pRes, num);
	nxCore::tMem<cxVec>::free(pSegs, numSegVecs);
	nxCore::tMem<cxVec>::free(pTris, numTriVecs);
}

static void f3_add_test(OGLSys::CL::Context ctx, OGLSys::CL::Queue que) {
	if (!ctx) return;
	if (!que) return;
	OGLSys::CL::Kernel kern = OGLSys::CL::create_kernel_from_src(ctx, s_f3_add_kern, "f3_add_kern");
	if (!kern) return;
	auto resBuf = f32x3_out_mem(ctx, 2);
	if (resBuf.valid()) {
		OGLSys::CL::set_kernel_buffer_arg(kern, 0, resBuf.mBuf);
		OGLSys::CL::set_kernel_float3_arg(kern, 1, 0.1f, 0.2f, 0.3f);
		OGLSys::CL::set_kernel_float3_arg(kern, 2, 1.0f, 2.0f, 3.0f);
		OGLSys::CL::exec_kernel(que, kern, 1);
		resBuf.force_update(que);
		OGLSys::CL::finish_queue(que);
		nxCore::dbg_msg("[CL f3 +] %f, %f %f\n", resBuf.mpMem->x, resBuf.mpMem->y, resBuf.mpMem->z);
		resBuf.reset();
	}
	OGLSys::CL::release_kernel(kern);
}

static void dot_test(OGLSys::CL::Context ctx, OGLSys::CL::Queue que, const int numElems, const int numVecs) {
	if (!ctx) return;
	if (!que) return;
	const char* pOpts = nullptr;
	OGLSys::CL::Device dev = OGLSys::CL::device_from_context(ctx);
	size_t numSrcItems = numElems * numVecs;
	size_t numDstItems = numVecs;
	float* pVec1CPU = (float*)nxCore::mem_alloc(numSrcItems * sizeof(float));
	float* pVec2CPU = (float*)nxCore::mem_alloc(numSrcItems * sizeof(float));
	float* pResCPU = (float*)nxCore::mem_alloc(numDstItems * sizeof(float));
	auto vec1Acc = f32_in_mem(ctx, numSrcItems);
	auto vec2Acc = f32_in_mem(ctx, numSrcItems);
	auto resAcc = f32_out_mem(ctx, numDstItems);
	bool bufsOk = vec1Acc.valid() && vec2Acc.valid() && resAcc.valid();
	if (!bufsOk) {
		vec1Acc.reset();
		vec2Acc.reset();
		resAcc.reset();
		return;
	}
	OGLSys::CL::Kernel kern = OGLSys::CL::create_kernel_from_src(ctx, s_dot_kern, "dot_kern", pOpts);
	if (kern) {
		OGLSys::CL::set_kernel_buffer_arg(kern, 0, resAcc.mBuf);
		OGLSys::CL::set_kernel_buffer_arg(kern, 1, vec1Acc.mBuf);
		OGLSys::CL::set_kernel_buffer_arg(kern, 2, vec2Acc.mBuf);
		OGLSys::CL::set_kernel_int_arg(kern, 3, numElems);
		float scl = nxCalc::rcp0(float(numVecs * numElems));
		for (int i = 0; i < numVecs; ++i) {
			int offs = i * numElems;
			float base = float(offs + 1);
			for (int j = 0; j < numElems; ++j) {
				float frac = (float(j) / float(numElems));
				float val = base + frac;
				val *= (i & 1) ? -scl : scl;
				pVec1CPU[offs + j] = val;
				vec1Acc.mpMem[offs + j] = val;
				val = base*2.0f + frac;
				val *= (i & 2) ? -scl : scl;
				pVec2CPU[offs + j] = val;
				vec2Acc.mpMem[offs + j] = val;
			}
		}
		vec1Acc.update(que);
		vec2Acc.update(que);

		/* warm up */
		OGLSys::CL::exec_kernel(que, kern, 1);
		resAcc.force_update(que);

		OGLSys::CL::Event evt;
		OGLSys::CL::exec_kernel(que, kern, numVecs, &evt);
		OGLSys::CL::flush_queue(que);
		double t0 = nxSys::time_micros();
		cpu_dot(pResCPU, pVec1CPU, pVec2CPU, numElems, numVecs);
		double dtCPU = nxSys::time_micros() - t0;
		int nwait = 0;
		while (true) {
			if (OGLSys::CL::event_ck_complete(evt)) {
				break;
			}
			//nxSys::sleep_millis(1);
			++nwait;
		}
		resAcc.update(que);//---------
		double dt = nxSys::time_micros() - t0;
		nxCore::dbg_msg("[CL dot] dt: %.3f micros (CPU: %.3f)\n", dt, dtCPU);
		OGLSys::CL::release_event(evt);
		float maxDiff = 0.0f;
		for (int i = 0; i < numVecs; ++i) {
			maxDiff = nxCalc::max(maxDiff, ::mth_fabsf(pResCPU[i] - resAcc.mpMem[i]));
			bool eq = nxCore::f32_almost_eq(pResCPU[i], resAcc.mpMem[i]);
			if (!eq) {
				nxCore::dbg_msg("[CL dot] CPU/Acc mismatch @ %d\n", i);
			}
		}
		nxCore::dbg_msg("[CL dot] CPU/Acc max diff: %.12f\n", maxDiff);
		nxCore::dbg_msg("# wait: %d\n", nwait);
	}

	if (OGLSys::CL::device_supports_fp16(dev)) {
		OGLSys::CL::Kernel kernHalf = OGLSys::CL::create_kernel_from_src(ctx, s_dot_kern_half, "dot_kern_half", pOpts);
		if (kernHalf) {
			auto vec1AccHalf = f16_in_mem(ctx, numSrcItems);
			auto vec2AccHalf = f16_in_mem(ctx, numSrcItems);
			auto resAccHalf = f16_out_mem(ctx, numDstItems);
			bufsOk = vec1AccHalf.valid() && vec2AccHalf.valid() && resAccHalf.valid();
			if (bufsOk) {
				OGLSys::CL::set_kernel_buffer_arg(kernHalf, 0, resAccHalf.mBuf);
				OGLSys::CL::set_kernel_buffer_arg(kernHalf, 1, vec1AccHalf.mBuf);
				OGLSys::CL::set_kernel_buffer_arg(kernHalf, 2, vec2AccHalf.mBuf);
				OGLSys::CL::set_kernel_int_arg(kernHalf, 3, numElems);
				for (size_t i = 0; i < numSrcItems; ++i) {
					vec1AccHalf.mpMem[i].set(vec1Acc.mpMem[i]);
				}
				for (size_t i = 0; i < numSrcItems; ++i) {
					vec2AccHalf.mpMem[i].set(vec2Acc.mpMem[i]);
				}
				vec1AccHalf.update(que);
				vec2AccHalf.update(que);
				OGLSys::CL::Event evtHalf;
				OGLSys::CL::exec_kernel(que, kernHalf, numVecs, &evtHalf);
				OGLSys::CL::flush_queue(que);
				double t0 = nxSys::time_micros();
				cpu_dot(pResCPU, pVec1CPU, pVec2CPU, numElems, numVecs);
				double dtCPU = nxSys::time_micros() - t0;
				int nwait = 0;
				while (true) {
					if (OGLSys::CL::event_ck_complete(evtHalf)) {
						break;
					}
					//nxSys::sleep_millis(1);
					++nwait;
				}
				resAccHalf.update(que);
				double dt = nxSys::time_micros() - t0;
				nxCore::dbg_msg("[CL dot fp16] dt: %f micros (CPU: %f)\n", dt, dtCPU);
				float maxDiff = 0.0f;
				for (int i = 0; i < numVecs; ++i) {
					maxDiff = nxCalc::max(maxDiff, ::mth_fabsf(pResCPU[i] - resAccHalf.mpMem[i].get()));
					float accRes = resAccHalf.mpMem[i].get();
					bool eq = nxCore::f32_almost_eq(pResCPU[i], accRes, 0.5f);
					if (!eq) {
						nxCore::dbg_msg("[CL dot fp16] CPU/Acc mismatch @ %d (%.12f != %.12f)\n", i, pResCPU[i], accRes);
					}
				}
				nxCore::dbg_msg("[CL dot fp16] CPU/Acc max diff: %.12f\n", maxDiff);
				nxCore::dbg_msg("# wait fp16: %d\n", nwait);
				OGLSys::CL::release_event(evtHalf);
			}
			OGLSys::CL::release_kernel(kernHalf);
			vec1AccHalf.reset();
			vec2AccHalf.reset();
			resAccHalf.reset();
		}
	}

	if (OGLSys::CL::device_supports_fp64(dev)) {
		OGLSys::CL::Kernel kernDbl = OGLSys::CL::create_kernel_from_src(ctx, s_dot_kern_dbl, "dot_kern_dbl", pOpts);
		if (kernDbl) {
			auto vec1AccDbl = f64_in_mem(ctx, numSrcItems);
			auto vec2AccDbl = f64_in_mem(ctx, numSrcItems);
			auto resAccDbl = f64_out_mem(ctx, numDstItems);
			bufsOk = vec1AccDbl.valid() && vec2AccDbl.valid() && resAccDbl.valid();
			if (bufsOk) {
				OGLSys::CL::set_kernel_buffer_arg(kernDbl, 0, resAccDbl.mBuf);
				OGLSys::CL::set_kernel_buffer_arg(kernDbl, 1, vec1AccDbl.mBuf);
				OGLSys::CL::set_kernel_buffer_arg(kernDbl, 2, vec2AccDbl.mBuf);
				OGLSys::CL::set_kernel_int_arg(kernDbl, 3, numElems);
				for (size_t i = 0; i < numSrcItems; ++i) {
					vec1AccDbl.mpMem[i] = vec1Acc.mpMem[i];
				}
				for (size_t i = 0; i < numSrcItems; ++i) {
					vec2AccDbl.mpMem[i] = vec2Acc.mpMem[i];
				}
				vec1AccDbl.update(que);
				vec2AccDbl.update(que);
				OGLSys::CL::Event evtDbl;
				OGLSys::CL::exec_kernel(que, kernDbl, numVecs, &evtDbl);
				OGLSys::CL::flush_queue(que);
				double t0 = nxSys::time_micros();
				cpu_dot(pResCPU, pVec1CPU, pVec2CPU, numElems, numVecs);
				double dtCPU = nxSys::time_micros() - t0;
				int nwait = 0;
				while (true) {
					if (OGLSys::CL::event_ck_complete(evtDbl)) {
						break;
					}
					//nxSys::sleep_millis(1);
					++nwait;
				}
				resAccDbl.update(que);
				double dt = nxSys::time_micros() - t0;
				nxCore::dbg_msg("[CL dot fp64] dt: %f micros (CPU: %f)\n", dt, dtCPU);
				float maxDiff = 0.0f;
				for (int i = 0; i < numVecs; ++i) {
					maxDiff = nxCalc::max(maxDiff, ::mth_fabsf(pResCPU[i] - float(resAccDbl.mpMem[i])));
					bool eq = nxCore::f32_almost_eq(pResCPU[i], float(resAccDbl.mpMem[i]));
					if (!eq) {
						nxCore::dbg_msg("[CL dot fp64] CPU/Acc mismatch @ %d\n", i);
					}
				}
				nxCore::dbg_msg("[CL dot fp64] CPU/Acc max diff: %.12f\n", maxDiff);
				nxCore::dbg_msg("# wait fp64: %d\n", nwait);
				OGLSys::CL::release_event(evtDbl);
			}
			OGLSys::CL::release_kernel(kernDbl);
			vec1AccDbl.reset();
			vec2AccDbl.reset();
			resAccDbl.reset();
		}
	}

	OGLSys::CL::release_kernel(kern);
	vec1Acc.reset();
	vec2Acc.reset();
	resAcc.reset();
	nxCore::mem_free(pVec1CPU);
	nxCore::mem_free(pVec2CPU);
	nxCore::mem_free(pResCPU);
}

} // namespace

XD_NOINLINE void test_ocl() {
	if (!OGLSys::CL::valid()) {
		return;
	}
	OGLSys::CL::PlatformList* pLst = OGLSys::CL::get_platform_list();
	if (pLst) {
		nxCore::dbg_msg("OpenCL platforms: %d\n", pLst->num);
		for (size_t i = 0; i < pLst->num; ++i) {
			nxCore::dbg_msg("==================================================================\n");
			nxCore::dbg_msg(" ---- [%d]: %s, %s\n", i, pLst->entries[i].pName, pLst->entries[i].pVendor);
			nxCore::dbg_msg("full: %s\n", pLst->entries[i].fullProfile ? "Yes" : "No");
			nxCore::dbg_msg("copr: %s\n", pLst->entries[i].coprFlg ? "Yes" : "No");
			nxCore::dbg_msg("vers: %s\n", pLst->entries[i].pVer);
			nxCore::dbg_msg("devs: %d (CPU: %d, GPU: %d, Acc: %d)\n",
				pLst->entries[i].numDevs, pLst->entries[i].numCPU, pLst->entries[i].numGPU, pLst->entries[i].numAcc);
			nxCore::dbg_msg("exts: %s\n", pLst->entries[i].pExts);
			nxCore::dbg_msg("max compute units: %d\n", OGLSys::CL::get_device_max_units(pLst->entries[i].defDev));
			nxCore::dbg_msg("max freq: %d\n", OGLSys::CL::get_device_max_freq(pLst->entries[i].defDev));
			nxCore::dbg_msg("global mem size: %.2f KB\n", OGLSys::CL::get_device_global_mem_size(pLst->entries[i].defDev));
			nxCore::dbg_msg("local mem: %s\n", OGLSys::CL::device_has_local_mem(pLst->entries[i].defDev) ? "Yes" : "No");
			nxCore::dbg_msg("local mem size: %.2f KB\n", OGLSys::CL::get_device_local_mem_size(pLst->entries[i].defDev));
			nxCore::dbg_msg("fast mem size: %.2f KB\n", OGLSys::CL::get_device_fast_mem_size(pLst->entries[i].defDev));
			nxCore::dbg_msg("fp16 support: %s\n", OGLSys::CL::device_supports_fp16(pLst->entries[i].defDev) ? "Yes" : "No");
			nxCore::dbg_msg("fp64 support: %s\n", OGLSys::CL::device_supports_fp64(pLst->entries[i].defDev) ? "Yes" : "No");
			nxCore::dbg_msg("byte-addressable: %s\n", OGLSys::CL::device_is_byte_addressable(pLst->entries[i].defDev) ? "Yes" : "No");
			OGLSys::CL::Context defCtx = OGLSys::CL::create_device_context(pLst->entries[i].defDev);
			nxCore::dbg_msg("def. context: %s\n", defCtx ? "OK" : "<invalid>");
			OGLSys::CL::print_device_exts(pLst->entries[i].defDev);
			OGLSys::CL::Queue que = OGLSys::CL::create_queue(defCtx);
			if (que) {
				f3_add_test(defCtx, que);
				seg_tri_test(defCtx, que, 1000);
				dot_test(defCtx, que, 100, 500);
				OGLSys::CL::release_queue(que);
			}
			OGLSys::CL::destroy_device_context(defCtx);
		}
		OGLSys::CL::free_platform_list(pLst);
	}
}
