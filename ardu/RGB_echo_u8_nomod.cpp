// Sketch uses 4022 bytes (14%) of program storage space. Maximum is 28672 bytes.
// Global variables use 179 bytes (6%) of dynamic memory, leaving 2381 bytes for local variables. Maximum is 2560 bytes.

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
        uint8_t idx = mIdx;
        idx = (idx >> 2) + (idx & 3);
        idx -= 3 * (idx >= 3);
        uint8_t scl = idx ? 10 : 1;
        uint16_t res = *pDst;
        res *= scl;
        res += val;
        if (res > 0xFF) {
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



RGB_U8 g_rgb;

void setup() {
  Serial.begin(115200);
  g_rgb.reset();
}

void loop() {
  while (Serial.available()) {
    uint8_t ch = Serial.read();
    if (ch == '\n') {
      Serial.println("**********");
      Serial.print("r: ");
      Serial.println(g_rgb.mR);
      Serial.print("g: ");
      Serial.println(g_rgb.mG);
      Serial.print("b: ");
      Serial.println(g_rgb.mB);
      g_rgb.reset();
    } else {
      g_rgb.parse_char(ch);
    }
  }
}

