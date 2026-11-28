// Runs hqx (faceloop/native/hqx, LGPL 2.1) on one 32-bit 0x00RRGGBB frame.
// LGPL 2.1 or later, like the library.
#include <mutex>
#include <stdint.h>
#include "hqx/hqx.h"

static std::once_flag table_ready;

extern "C" int hqx_scale(const uint32_t *src, uint32_t *dst, int width, int height, int scale)
{
    std::call_once(table_ready, hqxInit);
    if (scale == 2) hq2x_32(src, dst, width, height);
    else if (scale == 3) hq3x_32(src, dst, width, height);
    else if (scale == 4) hq4x_32(src, dst, width, height);
    else return -1;
    return 0;
}
