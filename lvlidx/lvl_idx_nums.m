clear all;

global _base_e_ = 1; % e if true, otherwise 10
global _bitmode_ = 0; % 0:32-bit, 1:40-bit

function res = enc_log(x)
	global _base_e_;
	if _base_e_
		res = log(x);
	else
		res = log10(x);
	end
end

function res = dec_exp(x)
	global _base_e_;
	if _base_e_
		res = exp(x);
	else
		res = 10^x;
	end
end

function n = lvl_idx_encode(x)
	if x >= 0 && x < 1
		n = x;
	else
		n = 1 + lvl_idx_encode(enc_log(x));
	end
end

function x = lvl_idx_decode(n)
	cnt = floor(n);
	x = n - cnt;
	for i = 1:cnt
		x = dec_exp(x);
	end
end

function [sgn, lvl, rlvl, rsd] = encode(x)
	sgn = x < 0;
	x = abs(x);
	enc0 = lvl_idx_encode(x);
	lvl = floor(enc0);
	r = enc0 - lvl;
	enc1 = 0;
	if r != 0
		enc1 = lvl_idx_encode(1.0 / r);
	end
	rlvl = floor(enc1);
	rsd = enc1 - rlvl;
end

function x = decode(sgn, lvl, rlvl, rsd)
	rdec = lvl_idx_decode(rlvl + rsd);
	r = rdec;
	if rdec != 0
		r = 1.0 / rdec;
	end
	x = lvl_idx_decode(lvl + r);
	if sgn
		x = -x;
	end
end

function [nbits, rbitsMax] = get_cfg()
	global _bitmode_;
	if _bitmode_ == 0
		nbits = 32;
		rbitsMax = 6;
	else
		nbits = 40;
		rbitsMax = 9;
	endif
end

% 32-bits
% 3x1-sgn, 1-mode, 3x{2|1}-lvl, 3x{5|6}-rsd, 3x2-rlvl
% 1 unused bit

% 40-bits
% 3x1-sgn, 1-mode, 3x{2|1}-lvl, 3x{8|9}-rsd, 3x2-rlvl
function res = sh_rgb_encode(rgb)
	[nbits, rbitsMax] = get_cfg();
	res = uint64(0);
	[sgn1, lvl1, rlvl1, rsd1] = encode(rgb(1));
	[sgn2, lvl2, rlvl2, rsd2] = encode(rgb(2));
	[sgn3, lvl3, rlvl3, rsd3] = encode(rgb(3));
	mode = 0;
	nrsd = rbitsMax;
	nlvl = 1;
	if lvl1 > 1 || lvl2 > 1 || lvl3 > 1
		mode = 1;
		nrsd -= 1;
		nlvl += 1;
	end
	scl = bitshift(1, nrsd) - 1;
	res = bitor(res, bitshift(real(sgn1), nbits-1));
	res = bitor(res, bitshift(real(sgn2), nbits-2));
	res = bitor(res, bitshift(real(sgn3), nbits-3));
	res = bitor(res, bitshift(mode, nbits-4));
	rlvl = [rlvl1, rlvl2, rlvl3];
	rsd = [rsd1, rsd2, rsd3];
	for i = 1:3
		if rlvl(i) > 3
			rlvl(i) = 0;
			rsd(i) = 0;
		end
	end
	rsd = uint64(floor(rsd .* scl));
	res = bitor(res, bitshift(bitor(bitor(lvl1, bitshift(lvl2,nlvl)), bitshift(lvl3,nlvl*2)), nbits-4 - nlvl*3));
	res = bitor(res, bitshift(bitor(bitor(rsd(1), bitshift(rsd(2), nrsd)), bitshift(rsd(3), nrsd*2)), 6));
	res = bitor(res, bitor(bitor(rlvl(1), bitshift(rlvl(2),2)), bitshift(rlvl(3),4)));
end

function res = sh_rgb_decode(enc)
	[nbits, rbitsMax] = get_cfg();
	res = zeros(1,3);
	sgn1 = bitand(enc, bitshift(1, nbits-1)) != 0;
	sgn2 = bitand(enc, bitshift(1, nbits-2)) != 0;
	sgn3 = bitand(enc, bitshift(1, nbits-3)) != 0;
	mode = bitand(enc, bitshift(1, nbits-4)) != 0;
	sgn = [sgn1, sgn2, sgn3];
	nrsd = rbitsMax;
	nlvl = 1;
	if mode
		nrsd -= 1;
		nlvl += 1;
	end
	slvl = nbits-4 - nlvl*3;
	scl = bitshift(1, nrsd) - 1;

	rlvl1 = bitand(enc, 3);
	rlvl2 = bitand(bitshift(enc, -2), 3);
	rlvl3 = bitand(bitshift(enc, -4), 3);
	rlvl = single([rlvl1, rlvl2, rlvl3]);

	bits = bitshift(enc, -6);
	mask = scl;
	rsd1 = bitand(bits, mask);
	rsd2 = bitand(bitshift(bits, -nrsd), mask);
	rsd3 = bitand(bitshift(bits, -nrsd*2), mask);
	rsd = single([rsd1, rsd2, rsd3]) / scl;

	bits = bitshift(enc, -slvl);
	mask = bitshift(1, nlvl) - 1;
	lvl1 = bitand(bits, mask);
	lvl2 = bitand(bitshift(bits, -nlvl), mask);
	lvl3 = bitand(bitshift(bits, -nlvl*2), mask);
	lvl = single([lvl1, lvl2, lvl3]);

	for i = 1:3
		res(i) = decode(sgn(i), lvl(i), rlvl(i), rsd(i));
	end
end



#en = lvl_idx_encode(12.345)
#lvl_idx_decode(en)

val = 12.345;#0.0001223; #3.3223
printf("val: %.12f\n", val);
[sgn, lvl, rlvl, rsd] = encode(val)
printf("decoded: %.12f\n", decode(sgn, lvl, rlvl, rsd));


#coefs = [0.01, 0.2, 2.3];
#coefs = [0.738, 0.6004, 0.4954];
#coefs = [100.0, 200.0, 300.0];
#coefs = [6.1072, 4.816, 3.7461];
coefs = [-0.0764, -0.0324, 0.0398]; # C024DEFE, C0503578BE
#coefs = [2.0082727, 1.22, 20.001];
coefs = [0.99, 0.5, 0.25];

enc = sh_rgb_encode(coefs);
dec2hex(enc)
dec = sh_rgb_decode(enc)

en = lvl_idx_encode(0xFFFFFFFF)
dec2hex(uint32(lvl_idx_decode(en)))


