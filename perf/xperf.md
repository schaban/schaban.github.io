***
|CPU/test| nnmul | isect | mkbvh | pano gen | pano SH |
|--------|--------|-----|--------|-----|-----|
| SiFive U74-MC @1.5GHz | 737 | 161 | 130 | 96 | 646 |
| Cortex-A35 @1.2GHz | 358 | 156 | 206  | 118 | 445  |
| Cortex-A53 @1.0GHz | 353  | 166 | 173  | 104 | 427  |
| Cortex-A15 @1.5GHz | 250 | 117  | 130  | 89 | 261  |
| Cortex-A15 @1.5GHz -mfpu=neon | 95 | 117  | 130  | 89 | 112  |
| Cortex-A57 @1.4GHz | 91 | 79 | 78 | 71 | 112 |
| Cortex-A72 @1.8GHz | 52 | 50  | 79  | 55 | 69 |
| i7-3770 @1.6GHz    | 49 | 59  | 48  | 44  | 73  |
| i7-3770 @1.6GHz -mavx | 28 | 53  | 48  | 44  | 73  |
| Apple M1 Max @ 2.5GHz| 16 | 15 | 24 | 16 | 22 |
***

<br>

native vs wasm
i7-3770 @1.6GHz:
| mode/test     | nnmul | isect | mkbvh | pano gen | pano SH |
|---------------|-------|-------|-------|----------|---------|
| native scalar | 151   | 58    |  49   | 44       | 185     |
| native sse    | 49    | 58    |  49   | 45       | 74      |
| native avx    | 29    | 54    |  48   | 45       | 74      |
| wasm scalar   | 233   | 55    |  76   | 116      | 186     |
| wasm simd128  | 31    | 56    |  76   | 120      | 94      |
| JS            | 993   | 459   | 641   | 3029     | 35194   |
| JS --jitless  | 63902 | 11340 | 5138  | 18808    | 52991   |
