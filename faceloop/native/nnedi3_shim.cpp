/*
 * Runs the nnedi3 filter (nnedi3.cpp, vendored unchanged from
 * dubhater/vapoursynth-nnedi3) on 8-bit planes without VapourSynth: a small
 * stand-in for the VapourSynth calls it makes. The filter's own create
 * function reads the parameters and nnedi3_weights.bin; its own getFrame
 * function does the work.
 *
 * Because it includes nnedi3.cpp, this file is under the GNU General Public
 * License version 2 or later (see LICENSE-GPL-2.txt).
 */
#define PACKAGE_VERSION "faceloop"
#include <vector>
#include "nnedi3.cpp"

#include <map>
#include <mutex>
#include <string>

struct VSFrameRef {
    int w, h;
    bool owned;
    std::vector<uint8_t> buf;
};
struct VSNodeRef { VSVideoInfo vi; };
struct VSMap {
    std::map<std::string, int64_t> ints;
    VSNodeRef * node = nullptr;
    std::string error;
};
struct VSPlugin { std::string path; };

static thread_local const VSFrameRef * g_src = nullptr;
static VSFormat g_fmt;
static VSPlugin g_plugin;

static void VS_CC sFreeFrame(const VSFrameRef * f) noexcept { if (f && f->owned) delete f; }
static void VS_CC sFreeNode(VSNodeRef *) noexcept {}
static VSFrameRef * VS_CC sNewVideoFrame(const VSFormat *, int w, int h, const VSFrameRef *, VSCore *) noexcept {
    return new VSFrameRef{ w, h, true, std::vector<uint8_t>(size_t(w) * h, 0) };
}
static VSPlugin * VS_CC sGetPluginById(const char *, VSCore *) noexcept { return &g_plugin; }
static const char * VS_CC sGetPluginPath(const VSPlugin * p) noexcept { return p->path.c_str(); }
static void * g_instance = nullptr;
static VSFilterGetFrame g_getFrame = nullptr;
static void VS_CC sCreateFilter(const VSMap *, VSMap *, const char *, VSFilterInit, VSFilterGetFrame getFrame, VSFilterFree,
                                int, int, void * instanceData, VSCore *) noexcept {
    g_instance = instanceData;
    g_getFrame = getFrame;
}
static void VS_CC sSetError(VSMap * map, const char * msg) noexcept { map->error = msg; }
static const VSFrameRef * VS_CC sGetFrameFilter(int, VSNodeRef *, VSFrameContext *) noexcept { return g_src; }
static void VS_CC sRequestFrameFilter(int, VSNodeRef *, VSFrameContext *) noexcept {}
static int VS_CC sStride(const VSFrameRef * f, int) noexcept { return f->w; }
static const uint8_t * VS_CC sRead(const VSFrameRef * f, int) noexcept { return f->buf.data(); }
static uint8_t * VS_CC sWrite(VSFrameRef * f, int) noexcept { return f->buf.data(); }
static const VSVideoInfo * VS_CC sGetVideoInfo(VSNodeRef * n) noexcept { return &n->vi; }
static void VS_CC sSetVideoInfo(const VSVideoInfo *, int, VSNode *) noexcept {}
static int VS_CC sWidth(const VSFrameRef * f, int) noexcept { return f->w; }
static int VS_CC sHeight(const VSFrameRef * f, int) noexcept { return f->h; }
static thread_local VSMap g_props;
static const VSMap * VS_CC sPropsRO(const VSFrameRef *) noexcept { g_props.ints.clear(); return &g_props; }
static VSMap * VS_CC sPropsRW(VSFrameRef *) noexcept { g_props.ints.clear(); return &g_props; }
static int VS_CC sNumElements(const VSMap * m, const char * key) noexcept { return m->ints.count(key) ? 1 : -1; }
static int64_t VS_CC sGetInt(const VSMap * m, const char * key, int index, int * err) noexcept {
    auto it = m->ints.find(key);
    if (it == m->ints.end() || index != 0) { if (err) *err = peUnset; return 0; }
    if (err) *err = 0;
    return it->second;
}
static VSNodeRef * VS_CC sGetNode(const VSMap * m, const char *, int, int * err) noexcept { if (err) *err = 0; return m->node; }
static int VS_CC sSetInt(VSMap *, const char *, int64_t, int) noexcept { return 0; }

static VSAPI makeApi() {
    VSAPI a;
    std::memset(&a, 0, sizeof(a));
    a.freeFrame = sFreeFrame; a.freeNode = sFreeNode; a.newVideoFrame = sNewVideoFrame;
    a.getPluginById = sGetPluginById; a.getPluginPath = sGetPluginPath; a.createFilter = sCreateFilter;
    a.setError = sSetError; a.getFrameFilter = sGetFrameFilter; a.requestFrameFilter = sRequestFrameFilter;
    a.getStride = sStride; a.getReadPtr = sRead; a.getWritePtr = sWrite; a.getVideoInfo = sGetVideoInfo;
    a.setVideoInfo = sSetVideoInfo; a.getFrameWidth = sWidth; a.getFrameHeight = sHeight;
    a.getFramePropsRO = sPropsRO; a.getFramePropsRW = sPropsRW; a.propNumElements = sNumElements;
    a.propGetInt = sGetInt; a.propGetNode = sGetNode; a.propSetInt = sSetInt;
    return a;
}
static const VSAPI g_api = makeApi();

#if defined(NNEDI3_ARM)
void getCPUFeatures(CPUFeatures * f) {  // every Apple Silicon (arm64) chip has NEON
    std::memset(f, 0, sizeof(*f));
    f->neon = 1;
}
#endif

static std::mutex g_lock;
static std::string g_key;
static VSNodeRef g_node;

/* Set up the filter for planes of w x h with these settings (once per change).
   Returns 0, or 1 with the filter's error in err (buffer of errlen). */
extern "C" int nnedi3_setup(const char * weights_dir, int w, int h, int nsize, int nns, int qual, int etype, int pscrn,
                            char * err, int errlen) {
    std::lock_guard<std::mutex> lk(g_lock);
    const std::string key = std::string(weights_dir) + "|" + std::to_string(w) + "x" + std::to_string(h) + "|" +
                            std::to_string(nsize) + "," + std::to_string(nns) + "," + std::to_string(qual) + "," +
                            std::to_string(etype) + "," + std::to_string(pscrn);
    if (key == g_key && g_instance)
        return 0;
    std::memset(&g_fmt, 0, sizeof(g_fmt));
    g_fmt.colorFamily = cmGray; g_fmt.sampleType = stInteger; g_fmt.bitsPerSample = 8; g_fmt.bytesPerSample = 1;
    g_fmt.numPlanes = 1;
    std::memset(&g_node.vi, 0, sizeof(g_node.vi));
    g_node.vi.format = &g_fmt; g_node.vi.width = w; g_node.vi.height = h; g_node.vi.numFrames = 1;
    g_node.vi.fpsNum = 1; g_node.vi.fpsDen = 1;
    g_plugin.path = std::string(weights_dir) + "/libnnedi3.so";
    VSMap in, out;
    in.node = &g_node;
    in.ints = { { "field", 1 }, { "dh", 1 }, { "nsize", nsize }, { "nns", nns }, { "qual", qual }, { "etype", etype },
                { "pscrn", pscrn } };
    g_instance = nullptr;
    nnedi3Create(&in, &out, nullptr, nullptr, &g_api);
    if (!out.error.empty() || !g_instance) {
        std::snprintf(err, errlen, "%s", out.error.empty() ? "nnedi3 setup failed" : out.error.c_str());
        g_key.clear();
        return 1;
    }
    g_key = key;
    return 0;
}

/* One plane: src w x h -> out w x 2h (originals on the even rows). Thread-safe after setup. */
extern "C" int nnedi3_plane(const uint8_t * src, int w, int h, uint8_t * out) {
    if (!g_instance || g_node.vi.width != w || g_node.vi.height != h)
        return 1;
    VSFrameRef s{ w, h, false, std::vector<uint8_t>(src, src + size_t(w) * h) };
    g_src = &s;
    void * inst = g_instance;
    const VSFrameRef * dst = g_getFrame(0, arAllFramesReady, &inst, nullptr, nullptr, nullptr, &g_api);
    g_src = nullptr;
    if (!dst)
        return 1;
    std::memcpy(out, dst->buf.data(), size_t(w) * 2 * h);
    sFreeFrame(dst);
    return 0;
}
