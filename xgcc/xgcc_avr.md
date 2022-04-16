***
1_get_arcs.sh
```
#!/bin/sh
mkdir -p arc
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
mkdir -p build

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
BIN_BU=build/avr-binutils
if [ ! -d "$BIN_BU" ]; then mkdir -p $BIN_BU; fi
BIN_GCC=build/avr-gcc
if [ ! -d "$BIN_GCC" ]; then mkdir -p $BIN_GCC; fi

AVR_PREFIX="--prefix=/usr/local/cross-avr"
AVR_OPTS="--target=avr --enable-threads=posix --with-system-zlib --with-isl --enable-__cxa_atexit --disable-libunwind-exceptions --enable-clocale=gnu --disable-libstdcxx-pch --disable-libssp --enable-plugin --disable-linker-build-id --enable-lto --enable-install-libiberty --with-linker-hash-style=gnu --with-gnu-ld --enable-gnu-indirect-function --disable-multilib --disable-werror --enable-checking=release --enable-default-pie --enable-default-ssp --enable-gnu-unique-object"

cd $BIN_BU
../binutils-*/configure $AVR_PREFIX $AVR_OPTS
make -j$(nproc)
make install
cd ../..

cd $BIN_GCC
../gcc-*/configure --enable-languages=c,c++ $AVR_PREFIX $AVR_OPTS
make all-gcc -j$(nproc)
make install-gcc
cd ../..

```


***
test_compile.sh
```
#!/bin/sh

CROSS_PATH="/usr/local/cross-avr"
CROSS_OPTS="-ffreestanding -nostartfiles -nostdlib -fno-builtin -fno-stack-protector -fno-PIC"

OPTI_LVL=-Os

_GCC_="$CROSS_PATH/bin/avr-gcc $CROSS_OPTS"
_ASM_="$_GCC_"
_GPP_="$CROSS_PATH/bin/avr-g++ $CROSS_OPTS -fno-exceptions -fno-rtti"
_CC_="$_GCC_ -std=c99 -Wall $OPTI_LVL"
_CXX_="$_GPP_ -std=c++11 -Wall $OPTI_LVL"
_LD_="$CROSS_PATH/bin/avr-ld -nostdlib"
_DISASM_="$CROSS_PATH/bin/avr-objdump -D"
_OBJCOPY_="$CROSS_PATH/bin/avr-objcopy"

$_CC_ -mmcu=atmega328p -c test.c -o test.o
$_DISASM_ test.o > test.txt
$_LD_ test.o -o test.out
$_OBJCOPY_ -O binary -R .eeprom test.out test.bin
$_DISASM_ -m avr -b binary test.bin > test_bin.txt
$_OBJCOPY_ -O ihex -R .eeprom test.out test.hex

```

***
test.c
```
#include <stdint.h>

#define _PORT_(addr) (*(volatile uint8_t*)(addr))
#define _DDRB_ _PORT_(0x24)
#define _PORTB_ _PORT_(0x25)

static void kilonops(const int n) {
	int i, j;
	for (i = 0; i < n; ++i) {
		for (j = 0; j < 1000; ++j) {
			__asm volatile("nop");
		}
	}
}

void main() {
	__asm volatile("sei");
	_DDRB_ |= 1 << 5;
	while (1) {
		_PORTB_ |= 1 << 5;
		kilonops(1000);
		_PORTB_ &= ~(1 << 5);
		kilonops(1000);
	}
}

```

***
(TODO) upload test.hex with avrdude

