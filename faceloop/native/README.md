# Native code

`EEDI2.cpp` is the VapourSynth EEDI2 filter (Kevin Stone's EEDI2 v0.9.2,
VapourSynth port by HolyWu), copied unchanged from
HomeOfVapourSynthEvolution/VapourSynth-EEDI2 at commit
36be83af2be47e31f92a33e56697e6d2bf866cec. It is licensed under the GNU GPL
version 2 or later (`LICENSE-GPL-2.txt`), and so is `eedi2_shim.cpp`, which
runs it on one plane without VapourSynth.

`nnedi3.cpp`, `simd_neon.c` and `cpufeatures.h` are the VapourSynth nnedi3
filter (Kevin Stone's NNEDI3, VapourSynth port by dubhater), copied
unchanged from dubhater/vapoursynth-nnedi3 at commit
82993ff21cf569776fc1a7e5bb60235c00bbeea3, GPL-2+, like `nnedi3_shim.cpp`,
which stands in for the VapourSynth calls the filter makes so its own
create and getFrame functions run. On Apple Silicon its NEON code is used
(the build falls back to plain C if that doesn't compile). Its weights
file (`nnedi3_weights.bin`, 13.5 MB, from the same commit, checked by
sha256) is downloaded to `weights/nnedi3/` the first time.

`VapourSynth.h` and `VSHelper.h` are the VapourSynth R54 API headers the
filter was written against (LGPL 2.1 and the header's own permissive
terms).

`hqx/` is the hqx library (Maxim Stepin's hq2x/hq3x/hq4x, C version by
Cameron Zemek and Francois Gannaz), copied unchanged from grom358/hqx at
commit a1c7d415f4bb7b947c4c2e47faf220b0148f4bac (`src/`, without the
command-line tool), LGPL 2.1 or later (`hqx/COPYING`), like `hqx_shim.cpp`,
which builds its colour table once and calls it.

faceloop compiles these into `_build/` the first time a filter is used, with
the system C++ compiler (`c++`, from Xcode's command line tools, which
Homebrew installs). Nothing else here is compiled.
