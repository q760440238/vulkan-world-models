use bytemuck::{Pod, Zeroable};
use std::borrow::Cow;
use wgpu::util::DeviceExt;

const N: usize = 1;
const C: usize = 2;
const D: usize = 4;
const H: usize = 2;
const W: usize = 2;
const K: usize = 3;
const PAD: usize = 1;
const HW: usize = H * W;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct ConvP { n: u32, cin: u32, cout: u32, h: u32, w: u32, k: u32, pad: u32, _p: u32 }
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct GNP { n: u32, c: u32, h: u32, w: u32, groups: u32, eps: f32, _p0: u32, _p1: u32 }
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct AddP { n: u32, _p0: u32, _p1: u32, _p2: u32 }
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct UpP { n: u32, c: u32, hin: u32, win: u32, hout: u32, wout: u32, _p0: u32, _p1: u32 }

const CONV_WGSL: &str = r#"
struct P { n: u32, cin: u32, cout: u32, h: u32, w: u32, k: u32, pad: u32, p: u32 };
@group(0) @binding(0) var<storage, read> xnew: array<f32>;
@group(0) @binding(1) var<storage, read> cache: array<f32>;
@group(0) @binding(2) var<storage, read> wt: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;
@group(0) @binding(5) var<uniform> p: P;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = p.n * p.cout * p.h * p.w;
    if (idx >= total) { return; }
    var t = idx;
    let ow = t % p.w; t = t / p.w;
    let oh = t % p.h; t = t / p.h;
    let co = t % p.cout;
    let bn = t / p.cout;
    var acc = bias[co];
    for (var ci = 0u; ci < p.cin; ci = ci + 1u) {
        for (var kd = 0u; kd < p.k; kd = kd + 1u) {
            for (var kh = 0u; kh < p.k; kh = kh + 1u) {
                let ih = i32(oh) + i32(kh) - i32(p.pad);
                if (ih < 0 || ih >= i32(p.h)) { continue; }
                for (var kw = 0u; kw < p.k; kw = kw + 1u) {
                    let iw = i32(ow) + i32(kw) - i32(p.pad);
                    if (iw < 0 || iw >= i32(p.w)) { continue; }
                    var xi = 0u;
                    if (kd == 2u) { xi = ((bn * p.cin + ci) * p.h + u32(ih)) * p.w + u32(iw); }
                    else { xi = (((bn * p.cin + ci) * 2u + kd) * p.h + u32(ih)) * p.w + u32(iw); }
                    let wi = (((co * p.cin + ci) * p.k + kd) * p.k + kh) * p.k + kw;
                    var xv = 0.0;
                    if (kd == 2u) { xv = xnew[xi]; } else { xv = cache[xi]; }
                    acc = acc + xv * wt[wi];
                }
            }
        }
    }
    y[idx] = acc;
}
"#;
const GN_WGSL: &str = r#"
struct P { n: u32, c: u32, h: u32, w: u32, groups: u32, eps: f32, p0: u32, p1: u32 };
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> gamma: array<f32>;
@group(0) @binding(2) var<storage, read> beta: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(4) var<uniform> p: P;
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tg = p.n * p.groups;
    if (gid.x >= tg) { return; }
    let bn = gid.x / p.groups;
    let g = gid.x % p.groups;
    let cg = p.c / p.groups;
    let c0 = g * cg;
    let spatial = p.h * p.w;
    let count = cg * spatial;
    var sum = 0.0; var sumsq = 0.0;
    for (var c = c0; c < c0 + cg; c = c + 1u) {
        for (var s = 0u; s < spatial; s = s + 1u) {
            let v = x[(bn * p.c + c) * spatial + s];
            sum = sum + v; sumsq = sumsq + v * v;
        }
    }
    let mean = sum / f32(count);
    let variance = sumsq / f32(count) - mean * mean;
    let inv = inverseSqrt(variance + p.eps);
    for (var c = c0; c < c0 + cg; c = c + 1u) {
        for (var s = 0u; s < spatial; s = s + 1u) {
            let idx = (bn * p.c + c) * spatial + s;
            let norm = (x[idx] - mean) * inv;
            let v = norm * gamma[c] + beta[c];
            y[idx] = v / (1.0 + exp(-v));
        }
    }
}
"#;
const UP_WGSL: &str = r#"
struct P { n: u32, c: u32, hin: u32, win: u32, hout: u32, wout: u32, p0: u32, p1: u32 };
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
@group(0) @binding(2) var<uniform> p: P;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = p.n * p.c * p.hout * p.wout;
    if (idx >= total) { return; }
    var t = idx;
    let ow = t % p.wout; t = t / p.wout;
    let oh = t % p.hout; t = t / p.hout;
    let co = t % p.c;
    let bn = t / p.c;
    let iw = ow / 2u;
    let ih = oh / 2u;
    y[idx] = x[((bn * p.c + co) * p.hin + ih) * p.win + iw];
}
"#;const ADD_WGSL: &str = r#"
struct P { n: u32, p0: u32, p1: u32, p2: u32 };
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> p: P;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.n) { return; }
    y[i] = a[i] + b[i];
}
"#;

fn full_conv(x: &[f32], w: &[f32], b: &[f32]) -> Vec<f32> {
    let mut y = vec![0.0f32; N * C * D * HW];
    for t in 0..D {
        for co in 0..C {
            for oh in 0..H {
                for ow in 0..W {
                    let mut acc = b[co];
                    for ci in 0..C {
                        for kd in 0..K {
                            let id = t as i32 + kd as i32 - (K as i32 - 1);
                            if id < 0 || id >= D as i32 { continue; }
                            for kh in 0..K {
                                let ih = oh as i32 + kh as i32 - PAD as i32;
                                if ih < 0 || ih >= H as i32 { continue; }
                                for kw in 0..K {
                                    let iw = ow as i32 + kw as i32 - PAD as i32;
                                    if iw < 0 || iw >= W as i32 { continue; }
                                    let xi = (((ci * D + id as usize) * H + ih as usize) * W) + iw as usize;
                                    let wi = (((co * C + ci) * K + kd) * K + kh) * K + kw;
                                    acc += x[xi] * w[wi];
                                }
                            }
                        }
                    }
                    y[((co * D + t) * H + oh) * W + ow] = acc;
                }
            }
        }
    }
    y
}
fn gn(x: &[f32], gamma: &[f32], beta: &[f32], p: &GNP) -> Vec<f32> {
    let spatial = (p.h * p.w) as usize;
    let cg = (p.c / p.groups) as usize;
    let mut y = vec![0.0f32; x.len()];
    for bn in 0..p.n as usize {
        for g in 0..p.groups as usize {
            let c0 = g * cg;
            let count = cg * spatial;
            let mut sum = 0.0f32; let mut sumsq = 0.0f32;
            for c in c0..c0 + cg {
                for s in 0..spatial {
                    let v = x[(bn * p.c as usize + c) * spatial + s];
                    sum += v; sumsq += v * v;
                }
            }
            let mean = sum / count as f32;
            let variance = sumsq / count as f32 - mean * mean;
            let inv = (variance + p.eps).sqrt().recip();
            for c in c0..c0 + cg {
                for s in 0..spatial {
                    let idx = (bn * p.c as usize + c) * spatial + s;
                    let norm = (x[idx] - mean) * inv;
                    let v = norm * gamma[c] + beta[c];
                    y[idx] = v / (1.0 + (-v).exp());
                }
            }
        }
    }
    y
}

fn upsample_frame(src: &[f32]) -> Vec<f32> {
    let h2 = H * 2; let w2 = W * 2;
    let mut y = vec![0.0f32; C * h2 * w2];
    for co in 0..C { for oh in 0..h2 { for ow in 0..w2 {
        let ih = oh / 2; let iw = ow / 2;
        y[(co * h2 + oh) * w2 + ow] = src[co * HW + ih * W + iw];
    }}}
    y
}
fn main() {
    let cp = ConvP { n: N as u32, cin: C as u32, cout: C as u32, h: H as u32, w: W as u32, k: K as u32, pad: PAD as u32, _p: 0 };
    let gp = GNP { n: N as u32, c: C as u32, h: H as u32, w: W as u32, groups: 1, eps: 1e-5, _p0: 0, _p1: 0 };
    let ap = AddP { n: (C * HW) as u32, _p0: 0, _p1: 0, _p2: 0 };
    let upp = UpP { n: N as u32, c: C as u32, hin: H as u32, win: W as u32, hout: (H * 2) as u32, wout: (W * 2) as u32, _p0: 0, _p1: 0 };
    let xn = N * C * D * HW;
    let wn = C * C * K * K * K;
    let mut x = vec![0.0f32; xn];
    let mut w1 = vec![0.0f32; wn];
    let mut w2 = vec![0.0f32; wn];
    let mut b1 = vec![0.0f32; C];
    let mut b2 = vec![0.0f32; C];
    let mut gamma = vec![0.0f32; C];
    let mut beta = vec![0.0f32; C];
    for i in 0..xn { x[i] = ((i * 13 % 71) as f32 - 35.0) / 19.0; }
    for i in 0..wn {
        w1[i] = ((i * 17 % 53) as f32 - 26.0) / 31.0;
        w2[i] = ((i * 19 % 47) as f32 - 23.0) / 29.0;
    }
    for i in 0..C {
        b1[i] = ((i * 7 % 11) as f32 - 5.0) * 0.01;
        b2[i] = ((i * 5 % 9) as f32 - 4.0) * 0.01;
        gamma[i] = 1.0 + ((i * 3 % 7) as f32 - 3.0) * 0.02;
        beta[i] = ((i * 11 % 13) as f32 - 6.0) * 0.01;
    }
    let cy1 = full_conv(&x, &w1, &b1);
    let mut gy = vec![0.0f32; C * D * HW];
    for t in 0..D {
        let mut frame = vec![0.0f32; C * HW];
        for ci in 0..C { frame[ci * HW..(ci + 1) * HW].copy_from_slice(&cy1[(ci * D + t) * HW..(ci * D + t + 1) * HW]); }
        let g = gn(&frame, &gamma, &beta, &gp);
        for ci in 0..C { gy[(ci * D + t) * HW..(ci * D + t + 1) * HW].copy_from_slice(&g[ci * HW..(ci + 1) * HW]); }
    }
    let cy2 = full_conv(&gy, &w2, &b2);
    let expect: Vec<f32> = x.iter().zip(cy2.iter()).map(|(a, b)| a + b).collect();
    let mut expect_up = vec![0.0f32; C * D * 4 * HW];
    for t in 0..D {
        let mut frame = vec![0.0f32; C * HW];
        for ci in 0..C { frame[ci * HW..(ci + 1) * HW].copy_from_slice(&expect[(ci * D + t) * HW..(ci * D + t + 1) * HW]); }
        let u = upsample_frame(&frame);
        for ci in 0..C { expect_up[(ci * D + t) * 4 * HW..(ci * D + t + 1) * 4 * HW].copy_from_slice(&u[ci * 4 * HW..(ci + 1) * 4 * HW]); }
    }

    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor { backends: wgpu::Backends::VULKAN, ..Default::default() });
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance, compatible_surface: None, force_fallback_adapter: false,
    })).expect("no Vulkan adapter");
    let info = adapter.get_info();
    println!("adapter: {} | backend: {:?}", info.name, info.backend);
    assert_eq!(info.backend, wgpu::Backend::Vulkan);
    let mut limits = wgpu::Limits::downlevel_defaults();
    limits.max_storage_buffers_per_shader_stage = 8;
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("stream-resblock"), required_features: wgpu::Features::empty(),
        required_limits: limits, ..Default::default()
    })).expect("device");
    let mk = |data: &[f32]| device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None, contents: bytemuck::cast_slice(data), usage: wgpu::BufferUsages::STORAGE,
    });
    let conv_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: None, source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(CONV_WGSL)) });
    let gn_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: None, source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(GN_WGSL)) });
    let add_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: None, source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(ADD_WGSL)) });
    let conv_pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: None, layout: None, module: &conv_mod, entry_point: Some("main"), compilation_options: Default::default(), cache: None });
    let gn_pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: None, layout: None, module: &gn_mod, entry_point: Some("main"), compilation_options: Default::default(), cache: None });
    let add_pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: None, layout: None, module: &add_mod, entry_point: Some("main"), compilation_options: Default::default(), cache: None });
    let up_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: None, source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(UP_WGSL)) });
    let up_pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: None, layout: None, module: &up_mod, entry_point: Some("main"), compilation_options: Default::default(), cache: None });
    let w1_buf = mk(&w1); let w2_buf = mk(&w2); let b1_buf = mk(&b1); let b2_buf = mk(&b2);
    let gamma_buf = mk(&gamma); let beta_buf = mk(&beta);
    let cp_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::bytes_of(&cp), usage: wgpu::BufferUsages::UNIFORM });
    let gp_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::bytes_of(&gp), usage: wgpu::BufferUsages::UNIFORM });
    let ap_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::bytes_of(&ap), usage: wgpu::BufferUsages::UNIFORM });

    let mut c1 = vec![0.0f32; C * 2 * HW];
    let c2_zeros = vec![0.0f32; C * 2 * HW];
    let mut cache2_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None, contents: bytemuck::cast_slice(&c2_zeros),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
    });
    let mut streamed = Vec::with_capacity(C * D * HW);
    for t in 0..D {
        let mut frame = vec![0.0f32; C * HW];
        for ci in 0..C { frame[ci * HW..(ci + 1) * HW].copy_from_slice(&x[(ci * D + t) * HW..(ci * D + t + 1) * HW]); }
        let frame_buf = mk(&frame); let c1_buf = mk(&c1);
        let y1 = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (C * HW * 4) as u64, usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false });
        let c1_bg = device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &conv_pipe.get_bind_group_layout(0), entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: frame_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: c1_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: w1_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: b1_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: y1.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: cp_buf.as_entire_binding() },
        ]});
        let y2 = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (C * HW * 4) as u64, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
        let g_bg = device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &gn_pipe.get_bind_group_layout(0), entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: y1.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: gamma_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: beta_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: y2.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: gp_buf.as_entire_binding() },
        ]});
        let y3 = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (C * HW * 4) as u64, usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false });
        let c2_bg = device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &conv_pipe.get_bind_group_layout(0), entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: y2.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: cache2_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: w2_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: b2_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: y3.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: cp_buf.as_entire_binding() },
        ]});
        let out = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (C * HW * 4) as u64, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
        let up_buf = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (C * 4 * HW * 4) as u64, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
        let up_rb = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (C * 4 * HW * 4) as u64, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
        let up_p = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::bytes_of(&upp), usage: wgpu::BufferUsages::UNIFORM });
        let a_bg = device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &add_pipe.get_bind_group_layout(0), entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: frame_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: y3.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: out.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: ap_buf.as_entire_binding() },
        ]});
        let up_bg = device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &up_pipe.get_bind_group_layout(0), entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: out.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: up_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: up_p.as_entire_binding() },
        ]});
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            p.set_pipeline(&conv_pipe); p.set_bind_group(0, &c1_bg, &[]); p.dispatch_workgroups(1, 1, 1);
        }
        {
            let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            p.set_pipeline(&gn_pipe); p.set_bind_group(0, &g_bg, &[]); p.dispatch_workgroups(1, 1, 1);
        }
        {
            let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            p.set_pipeline(&conv_pipe); p.set_bind_group(0, &c2_bg, &[]); p.dispatch_workgroups(1, 1, 1);
        }
        {
            let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            p.set_pipeline(&add_pipe); p.set_bind_group(0, &a_bg, &[]); p.dispatch_workgroups(1, 1, 1);
        }
        {
            let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            p.set_pipeline(&up_pipe); p.set_bind_group(0, &up_bg, &[]); p.dispatch_workgroups(1, 1, 1);
        }
        let next_cache = device.create_buffer(&wgpu::BufferDescriptor {
            label: None, size: (C * 2 * HW * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        for ci in 0..C {
            let base = (ci * 2 * HW * 4) as u64;
            let hw = (HW * 4) as u64;
            enc.copy_buffer_to_buffer(&cache2_buf, base + hw, &next_cache, base, hw);
            enc.copy_buffer_to_buffer(&y2, (ci * HW * 4) as u64, &next_cache, base + hw, hw);
        }
        cache2_buf = next_cache;
        enc.copy_buffer_to_buffer(&up_buf, 0, &up_rb, 0, (C * 4 * HW * 4) as u64);
        queue.submit(Some(enc.finish()));


        let s2 = up_rb.slice(..);
        s2.map_async(wgpu::MapMode::Read, |r| r.expect("map"));
        device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
        let d2 = s2.get_mapped_range();
        let got: &[f32] = bytemuck::cast_slice(&d2);
        let mut ferr = 0.0f32;
        for co in 0..C { for s in 0..4 * HW {
            let e = expect_up[(co * D + t) * 4 * HW + s];
            ferr = ferr.max((got[co * 4 * HW + s] - e).abs());
        }}
        println!("stream decoder frame {t}: frame_err={ferr:.3e}");
        streamed.extend_from_slice(got);
        drop(d2);
        up_rb.unmap();
        for bn in 0..N { for ci in 0..C {
            let base = (bn * C + ci) * 2 * HW;
            let old1 = c1[base + HW..base + 2 * HW].to_vec();
            c1[base..base + HW].copy_from_slice(&old1);
            c1[base + HW..base + 2 * HW].copy_from_slice(&frame[(bn * C + ci) * HW..(bn * C + ci + 1) * HW]);

        }}
    }
    let mut reordered = vec![0.0f32; streamed.len()];
    for t in 0..D { for co in 0..C { for s in 0..4 * HW {
        reordered[(co * D + t) * 4 * HW + s] = streamed[(t * C + co) * 4 * HW + s];
    }}}
    let mut max_err = 0.0f32;
    for (a, b) in reordered.iter().zip(expect_up.iter()) { max_err = max_err.max((a - b).abs()); }
    println!("stream-decoder frames={D} k={K} max_abs_err={max_err:.3e}");
    assert!(max_err < 1e-5, "stream resblock mismatch");
    println!("VULKAN STREAM DECODER OK");
}

















