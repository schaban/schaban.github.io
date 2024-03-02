// Sketch uses 5626 bytes (19%) of program storage space. Maximum is 28672 bytes.
// Global variables use 193 bytes (7%) of dynamic memory, leaving 2367 bytes for local variables. Maximum is 2560 bytes.

struct RGB_U8 {
  uint8_t mR;
  uint8_t mG;
  uint8_t mB;
  uint8_t mIdx;
  String mBuf;

  void reset() {
    mR = 0;
    mG = 0;
    mB = 0;
    mIdx = 0;
    mBuf = "";
  }

  void parse_char(const char ch) {
    if (ch >= '0' && ch <= '9') {
      mBuf += ch;
    } else {
      uint8_t* pDst = nullptr;
      if (mIdx == 0) {
        pDst = &mR;
      } else if (mIdx == 1) {
        pDst = &mG;
      } else if (mIdx == 2) {
        pDst = &mB;
      }
      if (pDst) {
        int res = mBuf.toInt();
        if (res > 0xFF) {
          res = 0xFF;
        }
        *pDst = (uint8_t)res;
        mBuf = "";
        ++mIdx;
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
    g_rgb.parse_char(ch);
    if (ch == '\n') {
      Serial.println("=========");
      Serial.print("r: ");
      Serial.println(g_rgb.mR);
      Serial.print("g: ");
      Serial.println(g_rgb.mG);
      Serial.print("b: ");
      Serial.println(g_rgb.mB);
      g_rgb.reset();
    }
  }
}

