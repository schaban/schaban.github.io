template<typename T>
struct HostMem {
	OGLSys::CL::Context mCtx;
	T* mpMem;
	size_t mMemSize;
	size_t mNumItems;
	OGLSys::CL::Buffer mBuf;
	bool mIsFast;
	bool mIsOut;
	bool mIsSVM;
	bool mIsViv;
	bool mIsARM;
	bool mIsIntel;

	void init(OGLSys::CL::Context ctx, const size_t n, const bool outFlg, const bool fastFlg = false) {
		mCtx = ctx;
		mpMem = nullptr;
		mMemSize = 0;
		mNumItems = 0;
		mBuf = nullptr;
		mIsFast = false;
		mIsOut = outFlg;
		mIsSVM = false;
		mIsViv = false;
		mIsARM = false;
		mIsIntel = false;
		if (OGLSys::CL::valid() && mCtx && n > 0) {
			OGLSys::CL::Device dev = OGLSys::CL::device_from_context(mCtx);
			mIsViv = OGLSys::CL::device_is_vivante_gpu(dev);
			if (!mIsViv) {
				mIsARM = OGLSys::CL::device_is_arm_gpu(dev);
				if (!mIsARM) {
					mIsIntel = OGLSys::CL::device_is_intel_gpu(dev);
				}
			}
			int memSizeAlign = 64;
			int memAddrAlign = 256;
			if (mIsIntel) {
				memAddrAlign = 4096;
			}
			mNumItems = n;
			mMemSize = XD_ALIGN(mNumItems * sizeof(T), memSizeAlign);
			if (mMemSize > 0) {
				if (fastFlg) {
					mpMem = (T*)OGLSys::CL::alloc_fast(mCtx, mMemSize);
					if (mpMem) {
						mIsFast = true;
					}
				}
				if (!mpMem) {
					mpMem = (T*)OGLSys::CL::alloc_svm(mCtx, mMemSize);
					if (mpMem) {
						mIsSVM = true;
					} else {
						mpMem = (T*)nxCore::mem_alloc(mMemSize, mIsOut ? "CL:HostOut" : "CL:HostIn", memAddrAlign);
					}
				}
				if (mpMem) {
					if (mIsOut) {
						mBuf = OGLSys::CL::create_host_mem_out_buffer(mCtx, mpMem, mMemSize);
					} else {
						mBuf = OGLSys::CL::create_host_mem_in_buffer(mCtx, mpMem, mMemSize);
					}
				}
			}
		}
	}

	void init_in(OGLSys::CL::Context ctx, const size_t n, const bool fastFlg = false) { init(ctx, n, false, fastFlg); }
	void init_out(OGLSys::CL::Context ctx, const size_t n, const bool fastFlg = false) { init(ctx, n, true, fastFlg); }

	void reset() {
		if (valid()) {
			OGLSys::CL::release_buffer(mBuf);
			mBuf = nullptr;
			if (mpMem) {
				if (mIsSVM) {
					OGLSys::CL::free_svm(mCtx, mpMem);
				} else if (mIsFast) {
					OGLSys::CL::free_fast(mCtx, mpMem);
				} else {
					nxCore::mem_free(mpMem);
				}
			}
			mpMem = nullptr;
			mMemSize = 0;
			mNumItems = 0;
		}
	}

	bool valid() const { return !!mBuf; }

	void update(OGLSys::CL::Queue que) {
		if (mIsSVM) return;
		if (mIsFast) return;
		if (que && valid()) {
			if (mIsOut) {
				if (!mIsViv && !mIsIntel) {
					OGLSys::CL::update_host_mem_out_buffer(que, mBuf, mpMem, mMemSize);
				}
			} else {
				if (mIsViv || mIsARM) {
					OGLSys::CL::update_host_mem_in_buffer(que, mBuf, mpMem, mMemSize);
				}
			}
		}
	}

	void force_update(OGLSys::CL::Queue que) {
		if (que && valid()) {
			if (mIsOut) {
				OGLSys::CL::update_host_mem_out_buffer(que, mBuf, mpMem, mMemSize);
			} else {
				OGLSys::CL::update_host_mem_in_buffer(que, mBuf, mpMem, mMemSize);
			}
		}
	}

	static HostMem<T> create_in(OGLSys::CL::Context ctx, const size_t n, const bool fastFlg = false) {
		HostMem<T> mem;
		mem.init_in(ctx, n, fastFlg);
		return mem;
	}

	static HostMem<T> create_out(OGLSys::CL::Context ctx, const size_t n, const bool fastFlg = false) {
		HostMem<T> mem;
		mem.init_out(ctx, n, fastFlg);
		return mem;
	}
};

inline HostMem<float> f32_in_mem(OGLSys::CL::Context ctx, const size_t n, const bool fastFlg = false) {
	return HostMem<float>::create_in(ctx, n, fastFlg);
}

inline HostMem<float> f32_out_mem(OGLSys::CL::Context ctx, const size_t n, const bool fastFlg = false) {
	return HostMem<float>::create_out(ctx, n, fastFlg);
}

inline HostMem<xt_half> f16_in_mem(OGLSys::CL::Context ctx, const size_t n, const bool fastFlg = false) {
	return HostMem<xt_half>::create_in(ctx, n, fastFlg);
}

inline HostMem<xt_half> f16_out_mem(OGLSys::CL::Context ctx, const size_t n, const bool fastFlg = false) {
	return HostMem<xt_half>::create_out(ctx, n, fastFlg);
}

inline HostMem<double> f64_in_mem(OGLSys::CL::Context ctx, const size_t n, const bool fastFlg = false) {
	return HostMem<double>::create_in(ctx, n, fastFlg);
}

inline HostMem<double> f64_out_mem(OGLSys::CL::Context ctx, const size_t n, const bool fastFlg = false) {
	return HostMem<double>::create_out(ctx, n, fastFlg);
}

inline HostMem<xt_float4> f32x4_in_mem(OGLSys::CL::Context ctx, const size_t n, const bool fastFlg = false) {
	return HostMem<xt_float4>::create_in(ctx, n, fastFlg);
}

inline HostMem<xt_float4> f32x4_out_mem(OGLSys::CL::Context ctx, const size_t n, const bool fastFlg = false) {
	return HostMem<xt_float4>::create_out(ctx, n, fastFlg);
}
