#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include "net.h"
#include "gpu.h"

int main(int argc, char** argv) {
    if (argc < 8) { fprintf(stderr, "usage: %s model.param model.bin in.f32 out.f32 W H [cpu|vulkan]\n", argv[0]); return 2; }
    ncnn::Net net;
    const std::string mode = argc > 7 ? argv[7] : "cpu";
    const bool fp16 = (mode == "vulkan16");
    net.opt.use_vulkan_compute = (mode == "vulkan" || mode == "vulkan16");
    net.opt.use_fp16_packed = fp16;
    net.opt.use_fp16_storage = fp16;
    net.opt.use_fp16_arithmetic = fp16;
    net.opt.num_threads = 8;
    if (net.opt.use_vulkan_compute) {
        ncnn::create_gpu_instance();
        net.set_vulkan_device(0);
    }
    if (net.load_param(argv[1]) != 0) { fprintf(stderr, "load_param failed\n"); return 1; }
    if (net.load_model(argv[2]) != 0) { fprintf(stderr, "load_model failed\n"); return 1; }
    const int W = atoi(argv[5]), H = atoi(argv[6]);
    FILE* f = fopen(argv[3], "rb");
    if (!f) { perror("input"); return 1; }
    std::vector<float> buf((size_t)W * H * 3);
    if (fread(buf.data(), 4, buf.size(), f) != buf.size()) { fprintf(stderr, "short input\n"); return 1; }
    fclose(f);
    ncnn::Mat in(W, H, 3);
    for (int c = 0; c < 3; c++) {
        float* dst = in.channel(c);
        const float* src = buf.data() + (size_t)c * W * H;
        for (int i = 0; i < W * H; i++) dst[i] = src[i];
    }
    ncnn::Extractor ex = net.create_extractor();
    ex.input("in0", in);
    ncnn::Mat out;
    int ret = ex.extract("out0", out);
    if (ret != 0) { fprintf(stderr, "extract failed %d\n", ret); return 1; }
    fprintf(stderr, "out w=%d h=%d c=%d dims=%d\n", out.w, out.h, out.c, out.dims);
    FILE* g = fopen(argv[4], "wb");
    for (int c = 0; c < out.c; c++) fwrite(out.channel(c), 4, (size_t)out.w * out.h, g);
    fclose(g);
    return 0;
}

