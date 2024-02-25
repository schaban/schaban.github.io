struct RGB_U8 {
	uint8_t mR;
	uint8_t mG;
	uint8_t mB;
	uint8_t mIdx;

	void reset() {
		mR = 0;
		mG = 0;
		mB = 0;
		mIdx = 0;
	}

	void parse_char(const char ch) {
		if (ch >= '0' && ch <= '9') {
			mIdx &= 0x7F;
			uint8_t* pDst = nullptr;
			uint8_t val = ch - '0';
			if (mIdx < 3) {
				pDst = &mR;
			} else if (mIdx < 6) {
				pDst = &mG;
			} else if (mIdx < 9) {
				pDst = &mB;
			}
			if (pDst) {
				uint8_t idx = (mIdx % 3);
				uint8_t scl = idx ? 10 : 1;
				uint16_t res = *pDst;
				res *= scl;
				res += val;
				if (res >= 0xFF) {
					res = 0xFF;
				}
				*pDst = (uint8_t)res;
				mIdx += !!(idx < 2);
			}
		} else {
			if (!(mIdx & 0x80)) {
				mIdx = ((mIdx / 3) + 1)*3;
				mIdx |= 0x80;
			}
		}
	}
};

