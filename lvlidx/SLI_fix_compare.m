clear all;

function n = LI_encode(x)
	if x >= 0 && x < 1
		n = x;
	else
		n = 1 + LI_encode(log(x));
	end
end

function x = LI_decode(n)
	cnt = floor(n);
	x = n - cnt;
	for i = 1:cnt
		x = exp(x);
	end
end

function res = encode01(x, rbits)
	res = 0;
	if x > 0
		sli = LI_encode(1.0 / x);
		lvl = floor(sli);
		if lvl < 4 % 2-bits for level
			idx = sli - lvl;
			scl = bitshift(1, rbits) - 1;
			res = bitor( uint32(floor(idx * scl)), bitshift(lvl, rbits) );
		else
			disp("cut!!!!!!!!!!!");
		end
	end
end

function x = decode01(enc, rbits)
	scl = bitshift(1, rbits) - 1;
	lvl = single(bitand(bitshift(enc, -rbits), 3));
	idx = single(bitand(enc, scl)) / scl;
	x = LI_decode(lvl + idx);
	if x > 0
		x = 1.0 / x;
	end
end

function test()
	#data = sin([0:0.01:pi/2-0.1]) ./ 100;
	data = [0:0.01:0.999] ./ 100;
	n = length(data);
	fracBits = 7;
	idxBits = fracBits-2;
	fixScl = bitshift(1, fracBits)-1;
	fixData = single(uint32(floor(data.*fixScl))) ./ fixScl;
	sliEnc = uint32(zeros(1,n));
	for i = 1:n
		sliEnc(i) = encode01(data(i), idxBits);
	end
	sliData = zeros(1,n);
	for i = 1:n
		sliData(i) = decode01(sliEnc(i), idxBits);
	end
	plot([data', fixData', sliData']);
end

test();


