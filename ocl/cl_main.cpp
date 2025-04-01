/// OGLSYS_CL=1

#include "crosscore.hpp"
#include "oglsys.hpp"

void test_ocl();

static void tst_dbgmsg(const char* pMsg) {
	//::OutputDebugStringA(pMsg);
	::printf("%s", pMsg);
	::fflush(stdout);
}

static void init_sys() {
	sxSysIfc sysIfc;
	nxCore::mem_zero(&sysIfc, sizeof(sysIfc));
	sysIfc.fn_dbgmsg = tst_dbgmsg;
	nxSys::init(&sysIfc);
}

int main(int argc, char* argv[]) {
	nxApp::init_params(argc, argv);
	init_sys();
	OGLSys::CL::init();

	test_ocl();

	OGLSys::CL::reset();
	nxApp::reset();
	nxCore::mem_dbg();

	return 0;

}
