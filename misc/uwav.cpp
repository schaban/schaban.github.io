int uwav_sin(int cnt, int msk = 0xFF, int scl = 0xFF) {
	float x = (float)(cnt & msk) / (float)(msk + 1);
	float y = (sinf(x*M_PI*2.0f - M_PI/2.0f) + 1.0f) / 2.0f;
	return (int)(y * (float)scl) & scl;
}

int uwav_tri(int cnt, int msk = 0xFF, int scl = 0xFF) {
	float x = (float)(cnt & msk) / (float)(msk + 1);
	float y = (x > 0.5f ? 1.0f - x : x) * 2.0f;
	return (int)(y * (float)scl) & scl;
}

