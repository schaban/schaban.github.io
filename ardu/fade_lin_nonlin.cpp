struct GLOB {
  int pin_lin;
  int pin_nonlin;
  int val;
  int add;

  void init() {
    pin_lin = 9;
    pin_nonlin = 10;
    pinMode(pin_lin, OUTPUT);
    pinMode(pin_nonlin, OUTPUT);
    val = 0;
    add = 5;
  }

  void exec() {
    int lin = val;
    int nonlin = float(lin)*lin*(1.0f/255.0f);
    analogWrite(pin_lin, lin);
    analogWrite(pin_nonlin, nonlin);
    val += add;
    if (val >= 255) {
      val = 255;
      add = -add;
    } else if (val < 0) {
      val = 0;
      add = -add;
    }
  }
} G;


void setup() {
  Serial.begin(9600);
  G.init();
}


void loop() {
  G.exec();
  delay(30);
}
