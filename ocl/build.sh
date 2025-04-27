#!/bin/sh

SYS_NAME=`uname -s`

case $SYS_NAME in
	CYGWIN*)
		SYS_DEFS="-DWIN32 -DXD_TSK_NATIVE=1"
		SYS_LIBS="-lgdi32"
	;;
	Linux)
		SYS_DEFS="-DDUMMY_GL"
		SYS_LIBS="-lpthread -ldl"

	;;
esac


# put crosscore, OGLSys and cl_* sources into src

c++ $SYS_DEFS -DOGLSYS_CL=1 -Wno-deprecated-declarations -Iinc -Isrc `ls src/*.cpp` $SYS_LIBS -o cl_test $*
