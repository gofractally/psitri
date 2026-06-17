# gcart read-perf + memory vs start_ver / sal_art_ref — 2026-06-17

Conditions: macOS arm64 (Apple), clang++ -O3 -DNDEBUG -march=native, 128 B grain.
1,000,000 random keys (len 4..23, mt19937 seed 99), 88-byte value, shuffled point
reads (hits). gcart = raw gcart::tree (no txn/commit). start_ver/sal_art_ref =
full kv adb path (batched txns of 4096 + commit), reads via read cursor get().

Build:
  clang++ -std=gnu++23 -O3 -DNDEBUG -DXXH_INLINE_ALL -march=native \
    -I libraries/gcart/include -I libraries/sal2/include -I libraries/sal/include \
    -I ~/gofractally/ucc/include gbench.cpp build/libraries/sal/libsal.a -o gbench
  clang++ ... -I libraries/start_ver/include -I libraries/sal_art_ref/include \
    -I libraries/sal_art_ref/test -I libraries/adb/include ... cmp.cpp ... -o cmp

Results (1M random keys):
  gcart        insert 257 ns   hit 166 ns   rss 210 MB   (165 MB live, 173 B/key)
  start_ver    insert 340 ns   hit 162 ns   rss 506 MB
  sal_art_ref  insert 1566 ns  hit 563 ns   rss 573 MB

=> read: gcart == start_ver (166 vs 162), 3.4x faster than sal_art_ref.
=> memory: gcart 2.4x lower than start_ver, 2.7x lower than sal_art_ref.
