***
1_get_arcs.sh
```
#!/bin/sh
if [ ! -d "arc" ]; then mkdir -p arc; fi
GNUFTP=https://ftpmirror.gnu.org
cd arc
wget $GNUFTP/binutils/binutils-2.37.tar.xz
wget $GNUFTP/gcc/gcc-11.2.0/gcc-11.2.0.tar.xz
wget $GNUFTP/mpfr/mpfr-4.1.0.tar.xz
wget $GNUFTP/gmp/gmp-6.2.1.tar.xz
wget $GNUFTP/mpc/mpc-1.2.1.tar.gz
wget https://gcc.gnu.org/pub/gcc/infrastructure/isl-0.18.tar.bz2
wget https://gcc.gnu.org/pub/gcc/infrastructure/cloog-0.18.1.tar.gz
cd ..
```

***
2_prebuild.sh
```
#!/bin/sh
if [ ! -d "build" ]; then mkdir -p build; fi

for mod in binutils gcc mpfr gmp mpc isl cloog
do
	echo "Extracting $mod..."
	tar -xf `ls arc/$mod-* | head -1` -C build
done

echo "Setting up lib links for binutils..."
cd build/binutils-*
for lib in mpfr gmp mpc isl
do
	ln -s ../$lib-* $lib
done
cd ../..

echo "Setting up lib links for gcc..."
cd build/gcc-*
for lib in mpfr gmp mpc isl cloog
do
	ln -s ../$lib-* $lib
done
cd ../..
```

***
(sudo) 3_build.sh
```
#!/bin/sh
BIN_BU=build/aa64-binutils
if [ ! -d "$BIN_BU" ]; then mkdir -p $BIN_BU; fi
BIN_GCC=build/aa64-gcc
if [ ! -d "$BIN_GCC" ]; then mkdir -p $BIN_GCC; fi

XGCC_PREFIX="--prefix=/usr/local/xgcc_aarch64"
XGCC_OPTS="--target=aarch64-elf --enable-shared --enable-threads=posix --with-system-zlib --with-isl --enable-__cxa_atexit --disable-libunwind-exceptions --enable-clocale=gnu --disable-libstdcxx-pch --disable-libssp --enable-plugin --disable-linker-build-id --enable-lto --enable-install-libiberty --with-linker-hash-style=gnu --with-gnu-ld --enable-gnu-indirect-function --disable-multilib --disable-werror --enable-checking=release --enable-default-pie --enable-default-ssp --enable-gnu-unique-object"

cd $BIN_BU
../binutils-*/configure $XGCC_PREFIX $XGCC_OPTS
make -j$(nproc)
make install
cd ../..

cd $BIN_GCC
../gcc-*/configure --enable-languages=c,c++ $XGCC_PREFIX $XGCC_OPTS
make all-gcc -j$(nproc)
make install-gcc
cd ../..
```


***
```
#!/bin/sh

CROSS_PATH="/usr/local/xgcc_aarch64"
CROSS_OPTS="-ffreestanding -nostartfiles -nostdlib -fno-builtin -fno-stack-protector"

_GCC_="$CROSS_PATH/bin/aarch64-elf-gcc $CROSS_OPTS"
_ASM_="$_GCC_"
_GPP_="$CROSS_PATH/bin/aarch64-elf-g++ $CROSS_OPTS -fno-exceptions -fno-rtti"
_CC_="$_GCC_ -std=c99 -Wall -O3 -ffast-math -ftree-vectorize"
_CXX_="$_GPP_ -std=c++11 -Wall -O3 -ffast-math -ftree-vectorize"
_LD_="$CROSS_PATH/bin/aarch64-elf-ld -nostdlib"
_DISASM_="$CROSS_PATH/bin/aarch64-elf-objdump -D"
_OBJCOPY_="$CROSS_PATH/bin/aarch64-elf-objcopy"

#!/bin/sh
if [ ! -d "tmp" ]; then
	mkdir -p tmp
fi

$_ASM_ -c src/boot.S -o tmp/boot.o

$_CXX_ -flto -g -T link.ld -Wl,--hash-style=sysv tmp/boot.o src/main.cpp -o kernel8.elf
$_DISASM_ kernel8.elf > kernel8.txt

$_OBJCOPY_ -O binary kernel8.elf kernel8.img
$_DISASM_ -m aarch64 -b binary kernel8.img > kernel8.img.txt
```


***
link.ld
```
SECTIONS {
	. = 0x80000;
	.text : { KEEP(*(*.boot)) *(.text .text.* .gnu.linkonce.t*) }
	.rodata : { *(.rodata .rodata.* .gnu.linkonce.r*) }
	PROVIDE(_data = .);
	.data : { *(.data .data.* .gnu.linkonce.d*) }
	.bss (NOLOAD): {
		. = ALIGN(16);
		__bss_start = .;
		*(.bss .bss.*)
		*(COMMON)
		__bss_end = .;
	}
	.mbox : {
		. = ALIGN(16);
		__mbox_mem = .;
		. = . + 0x1000;
	}
	.stack : {
		. = ALIGN(16);
		__stack_space = .;
		. = . + 0x100000 * 4;
	}
	_end = .;

	/DISCARD/ : { *(.comment) *(.gnu*) *(.note*) *(.eh_frame*) }
}
```


