#define PUSH_IN_PIN 8
#define PUSH_OUT_PIN 9

#define PUSH_NOW 1
#define PUSH_OLD 2

uint8_t g_pushState = 0;
uint32_t g_pushCnt = 0;

void setup() {
  Serial.begin(115200);
  pinMode(PUSH_IN_PIN, INPUT);
  pinMode(PUSH_OUT_PIN, OUTPUT);
}

__attribute__((noinline)) static void write_push(uint8_t val) {
  digitalWrite(PUSH_OUT_PIN, val);
}

static void update_state() {
  int btn = digitalRead(PUSH_IN_PIN);
  g_pushState &= ~PUSH_OLD;
  g_pushState |= (g_pushState & PUSH_NOW) ? PUSH_OLD : 0;
  g_pushState &= ~PUSH_NOW;
  g_pushState |= btn ? PUSH_NOW : 0;
}

void loop() {
  if (g_pushCnt > 0) {
    --g_pushCnt;
    g_pushState = PUSH_OLD;
  } else {
    update_state();
    bool now = !!(g_pushState & PUSH_NOW);
    bool old = !!(g_pushState & PUSH_OLD);
    bool chg = (now ^ old);
    bool trg = (now & chg);
    if (trg) {
      Serial.println("!");
      write_push(1);
      g_pushCnt = 70000;
      write_push(0);
    }
  }
}
