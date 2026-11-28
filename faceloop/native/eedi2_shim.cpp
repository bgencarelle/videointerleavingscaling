/*
 * Runs the EEDI2 filter (EEDI2.cpp, vendored unchanged from
 * HomeOfVapourSynthEvolution/VapourSynth-EEDI2) on one 8-bit plane, without
 * VapourSynth: a stand-in for the few frame calls it makes, and its
 * per-clip setup from eedi2Create, then its process<uint8_t>.
 *
 * Because it includes EEDI2.cpp, this file is under the GNU General Public
 * License version 2 or later (see LICENSE-GPL-2.txt).
 */
#include "EEDI2.cpp"

#include <cstring>
#include <thread>
#include <vector>

struct VSFrameRef {  // opaque in the API; ours: one plane, tightly packed
    int w, h;
    std::vector<uint8_t> buf;
};

static int VS_CC fWidth(const VSFrameRef * f, int) { return f->w; }
static int VS_CC fHeight(const VSFrameRef * f, int) { return f->h; }
static int VS_CC fStride(const VSFrameRef * f, int) { return f->w; }
static const uint8_t * VS_CC fRead(const VSFrameRef * f, int) { return f->buf.data(); }
static uint8_t * VS_CC fWrite(VSFrameRef * f, int) { return f->buf.data(); }

static VSAPI makeApi() {
    VSAPI a;
    std::memset(&a, 0, sizeof(a));
    a.getFrameWidth = fWidth;
    a.getFrameHeight = fHeight;
    a.getStride = fStride;
    a.getReadPtr = fRead;
    a.getWritePtr = fWrite;
    return a;
}

extern "C" int eedi2_plane(const uint8_t * src, int w, int h, uint8_t * out, int field, int mthresh, int lthresh,
                           int vthresh, int estr, int dstr, int maxd, int nt, int pp) {
    if (w < 8 || h < 7 || maxd < 1 || maxd > 29 || pp < 0 || pp > 3 || field < 0 || field > 1)
        return 1;
    static const VSAPI api = makeApi();
    VSFormat fmt;
    std::memset(&fmt, 0, sizeof(fmt));
    fmt.colorFamily = cmGray;
    fmt.sampleType = stInteger;
    fmt.bitsPerSample = 8;
    fmt.bytesPerSample = 1;
    fmt.numPlanes = 1;
    VSVideoInfo vi;
    std::memset(&vi, 0, sizeof(vi));
    vi.format = &fmt;
    vi.width = w;
    vi.height = h;

    // eedi2Create, after argument parsing (map = 0: the doubled frame)
    EEDI2Data d{};
    d.vi = &vi;
    d.vi2 = vi;
    d.vi2.height *= 2;
    d.field = field;
    d.fieldS = field;
    d.mthresh = mthresh * mthresh;
    d.lthresh = lthresh;
    d.vthresh = vthresh * 81;
    d.estr = estr;
    d.dstr = dstr;
    d.maxd = maxd;
    d.map = 0;
    d.pp = pp;
    const int8_t limlut[33]{ 6, 6, 7, 7, 8, 8, 9, 9, 9, 10, 10, 11, 11, 12, 12, 12, 12, 12, 12, 12,
                             12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, -1, -1 };
    std::vector<int8_t> ll(limlut, limlut + 33);
    std::vector<int16_t> ll2(33);
    for (unsigned i = 0; i < 33; i++)
        ll2[i] = limlut[i];
    d.limlut = ll.data();
    d.limlut2 = ll2.data();
    d.nt4 = nt * 4;
    d.nt7 = nt * 7;
    d.nt8 = nt * 8;
    d.nt13 = nt * 13;
    d.nt19 = nt * 19;
    // eedi2GetFrame's per-thread buffers
    const auto id = std::this_thread::get_id();
    std::vector<int> cx2, cy2, cxy, tmpc;
    if (pp > 1) {
        cx2.resize(w * h); cy2.resize(w * h); cxy.resize(w * h); tmpc.resize(w * h);
    }
    d.cx2.emplace(id, pp > 1 ? cx2.data() : nullptr);
    d.cy2.emplace(id, pp > 1 ? cy2.data() : nullptr);
    d.cxy.emplace(id, pp > 1 ? cxy.data() : nullptr);
    d.tmpc.emplace(id, pp > 1 ? tmpc.data() : nullptr);

    auto frame = [](int fw, int fh) { return VSFrameRef{ fw, fh, std::vector<uint8_t>(size_t(fw) * fh, 0) }; };
    VSFrameRef s = frame(w, h), dst = frame(w, h), msk = frame(w, h), tmp = frame(w, h);
    VSFrameRef dst2 = frame(w, 2 * h), dst2M = frame(w, 2 * h), tmp2 = frame(w, 2 * h), tmp2_2 = frame(w, 2 * h),
               msk2 = frame(w, 2 * h);
    std::memcpy(s.buf.data(), src, size_t(w) * h);
    process<uint8_t>(&s, &dst, &msk, &tmp, &dst2, &dst2M, &tmp2, &tmp2_2, &msk2, unsigned(field), &d, nullptr, &api);
    std::memcpy(out, dst2.buf.data(), size_t(w) * 2 * h);
    return 0;
}
