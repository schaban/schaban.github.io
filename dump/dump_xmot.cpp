#include "crosscore.hpp"


static void dbgmsg_impl(const char* pMsg) {
	::fprintf(stderr, "%s", pMsg);
	::fflush(stderr);
}

static void init_sys() {
	sxSysIfc sysIfc;
	nxCore::mem_zero(&sysIfc, sizeof(sysIfc));
	sysIfc.fn_dbgmsg = dbgmsg_impl;
	nxSys::init(&sysIfc);
}

static void reset_sys() {
}


static void dump_xmot(FILE* pOut, sxMotionData* pMot) {
	int nframes = pMot->mFrameNum;
	uint32_t nnodes = pMot->mNodeNum;
	for (uint32_t inode = 0; inode < nnodes; ++inode) {
		const char* pNodeName = pMot->get_node_name(inode);
		if (pMot->get_q_track(inode)) {
			::fprintf(pOut, "-- %s: Quat track\n", pNodeName);
			for (int fno = 0; fno < nframes; ++fno) {
				cxQuat q = pMot->eval_quat(inode, fno);
				::fprintf(pOut, "%d: %f, %f, %f, %f \n",
				                 fno, q.x, q.y, q.z, q.w);
			}  
		}
		if (pMot->get_t_track(inode)) {
			::fprintf(pOut, "-- %s: Trans track\n", pNodeName);
			for (int fno = 0; fno < nframes; ++fno) {
				cxVec t = pMot->eval_pos(inode, fno);
				::fprintf(pOut, "%d: %f, %f, %f\n",
			                         fno, t.x, t.y, t.z);
			}
		}
	}
} 

int main(int argc, char* argv[]) {
	nxApp::init_params(argc, argv);
	init_sys();

	const char* pMotPath = nxApp::get_arg(0);
	if (pMotPath) {
		sxMotionData* pMot = nxData::load_as<sxMotionData>(pMotPath);
		if (pMot) {
			dump_xmot(stdout, pMot);
		} else {
			::fprintf(stderr, "Invalid file format.\n");
		}
	} else {
		::fprintf(stderr, "dump_xmot src.xmot\n");
	}

	nxApp::reset();
	reset_sys();
	return 0;
}
