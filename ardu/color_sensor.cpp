const int NSMPS = 16;

struct FILTER {
	int smps[NSMPS];
	int lvl;

	void init() {
		for (int i = 0; i < NSMPS; ++i) {
			smps[i] = 0;
		}
	}

	void sample(int pin) {
		for (int i = 1; i < NSMPS; ++i) {
			smps[i-1] = smps[i];
		}
		unsigned long puls = min(pulseIn(pin, HIGH), 1000UL);
		smps[NSMPS-1] = map(puls, 1000, 10, 0, 255);
		lvl = smps[0];
		for (int i = 1; i < NSMPS; ++i) {
			lvl += smps[i];
		}
		lvl /= NSMPS;
	}

};

static struct FARBA {
	int pin_s2;
	int pin_s3;
	int pin_readout;
	int pin_vis_R;
	int pin_vis_G;
	int pin_vis_B;
	FILTER R;
	FILTER G;
	FILTER B;
	FILTER L;

	void start() {
		Serial.begin(9600);
		pin_s2 = 5;
		pin_s3 = 6;
		pin_readout = 8;
		pin_vis_R = 3;
		pin_vis_G = 9;
		pin_vis_B = 11;
		pinMode(pin_s2, OUTPUT);
		pinMode(pin_s3, OUTPUT);
		pinMode(pin_readout, INPUT);
		pinMode(pin_vis_R, OUTPUT);
		pinMode(pin_vis_G, OUTPUT);
		pinMode(pin_vis_B, OUTPUT);
		R.init();
		G.init();
		B.init();
		L.init();
	}

	void normalize() {
		R.lvl = (long(R.lvl) * 255) / L.lvl;
		G.lvl = (long(G.lvl) * 255) / L.lvl;
		B.lvl = (long(B.lvl) * 255) / L.lvl;
	}

	void exec() {
		digitalWrite(pin_s2, LOW);
		digitalWrite(pin_s3, LOW);
		R.sample(pin_readout);

		digitalWrite(pin_s2, HIGH);
		digitalWrite(pin_s3, HIGH);
		G.sample(pin_readout);

		digitalWrite(pin_s2, LOW);
		digitalWrite(pin_s3, HIGH);
		B.sample(pin_readout);

		digitalWrite(pin_s2, HIGH);
		digitalWrite(pin_s3, LOW);
		L.sample(pin_readout);

		normalize();
	}

	void plot() {
		Serial.print(B.lvl); Serial.print(" ");
		Serial.print(R.lvl); Serial.print(" ");
		Serial.print(G.lvl); Serial.print(" ");
		Serial.println(L.lvl);
	}

	static void vis_sub(int pin, int val) {
		int v = powf(val, 2) / 255;
		analogWrite(pin, v);
	}

	void vis() {
		vis_sub(pin_vis_R, R.lvl);
		vis_sub(pin_vis_G, G.lvl);
 		vis_sub(pin_vis_B, B.lvl);
	}
} farba;


void setup() {
	farba.start();
}


void loop() {
	farba.exec();
	farba.plot();
	farba.vis();
	delay(30);
}
