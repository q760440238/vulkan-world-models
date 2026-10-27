use bytemuck::{Pod, Zeroable};
use std::borrow::Cow;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct ConvParams {
    n: u32, cin: u32, cout: u32,
    din: u32, hin: u32, win: u32,
    kd: u32, kh: u32, kw: u32,
    pad: u32, stride: u32,
    dout: u32, hout: u32, wout: u32,
    _p0: u32, _p1: u32,
}
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct GNParams { n: u32, c: u32, d: u32, h: u32, w: u32, groups: u32, eps: f32, _p: u32 }
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct UpParams {
    n: u32, c: u32, din: u32, hin: u32, win: u32,
    scale: u32, dout: u32, hout: u32, wout: u32,
    _p0: u32, _p1: u32, _p2: u32,
}

const CONV_WGSL: &str = r#"
struct P { n: u32, cin: u32, cout: u32, din: u32, hin: u32, win: u32, kd: u32, kh: u32, kw: u32,
           pad: u32, stride: u32, dout: u32, hout: u32, wout: u32, p0: u32, p1: u32 };
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(4) var<uniform> p: P;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = p.n * p.cout * p.dout * p.hout * p.wout;
    if (idx >= total) { return; }
    var t = idx;
    let ow = t % p.wout; t = t / p.wout;
    let oh = t % p.hout; t = t / p.hout;
    let od = t % p.dout; t = t / p.dout;
    let co = t % p.cout;
    let bn = t / p.cout;
    var acc = b[co];
    for (var ci = 0u; ci < p.cin; ci = ci + 1u) {
        for (var kd = 0u; kd < p.kd; kd = kd + 1u) {
            let id = i32(od * p.stride + kd) - i32(p.pad);
            if (id < 0 || id >= i32(p.din)) { continue; }
            for (var kh = 0u; kh < p.kh; kh = kh + 1u) {
                let ih = i32(oh * p.stride + kh) - i32(p.pad);
                if (ih < 0 || ih >= i32(p.hin)) { continue; }
                for (var kw = 0u; kw < p.kw; kw = kw + 1u) {
                    let iw = i32(ow * p.stride + kw) - i32(p.pad);
                    if (iw < 0 || iw >= i32(p.win)) { continue; }
                    let xidx = (((bn * p.cin + ci) * p.din + u32(id)) * p.hin + u32(ih)) * p.win + u32(iw);
                    let widx = (((co * p.cin + ci) * p.kd + kd) * p.kh + kh) * p.kw + kw;
                    acc = acc + x[xidx] * w[widx];
                }
            }
        }
    }
    y[idx] = acc;
}
"#;

const GN_WGSL: &str = r#"
struct P { n: u32, c: u32, d: u32, h: u32, w: u32, groups: u32, eps: f32, p: u32 };
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
    let spatial = p.d * p.h * p.w;
    let count = cg * spatial;
    var sum = 0.0;
    var sumsq = 0.0;
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
struct P { n: u32, c: u32, din: u32, hin: u32, win: u32, scale: u32, dout: u32, hout: u32, wout: u32,
           p0: u32, p1: u32, p2: u32 };
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
@group(0) @binding(2) var<uniform> p: P;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = p.n * p.c * p.dout * p.hout * p.wout;
    if (idx >= total) { return; }
    var t = idx;
    let ow = t % p.wout; t = t / p.wout;
    let oh = t % p.hout; t = t / p.hout;
    let od = t % p.dout; t = t / p.dout;
    let co = t % p.c;
    let bn = t / p.c;
    let iw = ow / p.scale;
    let ih = oh / p.scale;
    let id = od / p.scale;
    let src = (((bn * p.c + co) * p.din + id) * p.hin + ih) * p.win + iw;
    y[idx] = x[src];
}
"#;

fn cpu_conv(x: &[f32], w: &[f32], b: &[f32], p: &ConvParams) -> Vec<f32> {
    let total = (p.n * p.cout * p.dout * p.hout * p.wout) as usize;
    let mut y = vec![0.0f32; total];
    for idx in 0..total {
        let mut t = idx;
        let ow = t % p.wout as usize; t /= p.wout as usize;
        let oh = t % p.hout as usize; t /= p.hout as usize;
        let od = t % p.dout as usize; t /= p.dout as usize;
        let co = t % p.cout as usize;
        let bn = t / p.cout as usize;
        let mut acc = b[co];
        for ci in 0..p.cin as usize {
            for kd in 0..p.kd as usize {
                let id = od as i32 * p.stride as i32 + kd as i32 - p.pad as i32;
                if id < 0 || id >= p.din as i32 { continue; }
                for kh in 0..p.kh as usize {
                    let ih = oh as i32 * p.stride as i32 + kh as i32 - p.pad as i32;
                    if ih < 0 || ih >= p.hin as i32 { continue; }
                    for kw in 0..p.kw as usize {
                        let iw = ow as i32 * p.stride as i32 + kw as i32 - p.pad as i32;
                        if iw < 0 || iw >= p.win as i32 { continue; }
                        let xidx = (((bn * p.cin as usize + ci) * p.din as usize + id as usize) * p.hin as usize + ih as usize) * p.win as usize + iw as usize;
                        let widx = (((co * p.cin as usize + ci) * p.kd as usize + kd) * p.kh as usize + kh) * p.kw as usize + kw;
                        acc += x[xidx] * w[widx];
                    }
                }
            }
        }
        y[idx] = acc;
    }
    y
}
fn cpu_gn(x: &[f32], gamma: &[f32], beta: &[f32], p: &GNParams) -> Vec<f32> {
    let spatial = (p.d * p.h * p.w) as usize;
    let cg = (p.c / p.groups) as usize;
    let mut y = vec![0.0f32; x.len()];
    for bn in 0..p.n as usize {
        for g in 0..p.groups as usize {
            let c0 = g * cg;
            let count = cg * spatial;
            let mut sum = 0.0f32;
            let mut sumsq = 0.0f32;
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
fn cpu_up(x: &[f32], p: &UpParams) -> Vec<f32> {
    let mut y = vec![0.0f32; (p.n * p.c * p.dout * p.hout * p.wout) as usize];
    for idx in 0..y.len() {
        let mut t = idx;
        let ow = t % p.wout as usize; t /= p.wout as usize;
        let oh = t % p.hout as usize; t /= p.hout as usize;
        let od = t % p.dout as usize; t /= p.dout as usize;
        let co = t % p.c as usize;
        let bn = t / p.c as usize;
        let iw = ow / p.scale as usize;
        let ih = oh / p.scale as usize;
        let id = od / p.scale as usize;
        y[idx] = x[(((bn * p.c as usize + co) * p.din as usize + id) * p.hin as usize + ih) * p.win as usize + iw];
    }
    y
}

fn main() {
    let cp = ConvParams { n: 1, cin: 2, cout: 2, din: 2, hin: 2, win: 2,
                         kd: 3, kh: 3, kw: 3, pad: 1, stride: 1,
                         dout: 2, hout: 2, wout: 2, _p0: 0, _p1: 0 };
    let gp = GNParams { n: 1, c: 2, d: 2, h: 2, w: 2, groups: 1, eps: 1e-5, _p: 0 };
    let up = UpParams { n: 1, c: 2, din: 2, hin: 2, win: 2, scale: 2, dout: 4, hout: 4, wout: 4,
                        _p0: 0, _p1: 0, _p2: 0 };
    let xn = (cp.n * cp.cin * cp.din * cp.hin * cp.win) as usize;
    let wn = (cp.cout * cp.cin * cp.kd * cp.kh * cp.kw) as usize;
    let yn = (cp.n * cp.cout * cp.dout * cp.hout * cp.wout) as usize;
    let un = (up.n * up.c * up.dout * up.hout * up.wout) as usize;
    let mut x = vec![0.0f32; xn];
    let mut w = vec![0.0f32; wn];
    let mut b = vec![0.0f32; cp.cout as usize];
    let mut gamma = vec![0.0f32; gp.c as usize];
    let mut beta = vec![0.0f32; gp.c as usize];
    for i in 0..xn { x[i] = ((i * 13 % 71) as f32 - 35.0) / 19.0; }
    for i in 0..wn { w[i] = ((i * 17 % 53) as f32 - 26.0) / 31.0; }
    for i in 0..b.len() { b[i] = ((i * 7 % 11) as f32 - 5.0) * 0.01; }
    for i in 0..gamma.len() { gamma[i] = 1.0 + ((i * 5 % 9) as f32 - 4.0) * 0.02; }
    for i in 0..beta.len() { beta[i] = ((i * 3 % 7) as f32 - 3.0) * 0.01; }
    let conv_cpu = cpu_conv(&x, &w, &b, &cp);
    let gn_cpu = cpu_gn(&conv_cpu, &gamma, &beta, &gp);
    let expect = cpu_up(&gn_cpu, &up);

    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor { backends: wgpu::Backends::VULKAN, ..Default::default() });
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance, compatible_surface: None, force_fallback_adapter: false,
    })).expect("no Vulkan adapter");
    let info = adapter.get_info();
    println!("adapter: {} | backend: {:?}", info.name, info.backend);
    assert_eq!(info.backend, wgpu::Backend::Vulkan);

    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("vae-mini"), required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::downlevel_defaults(), ..Default::default()
    })).expect("device");

    let mk = |data: &[f32]| device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None, contents: bytemuck::cast_slice(data), usage: wgpu::BufferUsages::STORAGE,
    });
    let x_buf = mk(&x); let w_buf = mk(&w); let b_buf = mk(&b);
    let conv_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: None, size: (yn * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false,
    });
    let gamma_buf = mk(&gamma); let beta_buf = mk(&beta);
    let gn_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: None, size: (yn * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false,
    });
    let up_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: None, size: (un * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false,
    });
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: None, size: (un * 4) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false,
    });
    let conv_p = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None, contents: bytemuck::bytes_of(&cp), usage: wgpu::BufferUsages::UNIFORM,
    });
    let gn_p = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None, contents: bytemuck::bytes_of(&gp), usage: wgpu::BufferUsages::UNIFORM,
    });
    let up_p = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None, contents: bytemuck::bytes_of(&up), usage: wgpu::BufferUsages::UNIFORM,
    });

    let conv_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: None, source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(CONV_WGSL)) });
    let gn_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: None, source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(GN_WGSL)) });
    let up_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: None, source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(UP_WGSL)) });
    let conv_pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: None, layout: None, module: &conv_mod, entry_point: Some("main"), compilation_options: Default::default(), cache: None });
    let gn_pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: None, layout: None, module: &gn_mod, entry_point: Some("main"), compilation_options: Default::default(), cache: None });
    let up_pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: None, layout: None, module: &up_mod, entry_point: Some("main"), compilation_options: Default::default(), cache: None });

    let conv_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None, layout: &conv_pipe.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: x_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: w_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: b_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: conv_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: conv_p.as_entire_binding() },
        ],
    });
    let gn_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None, layout: &gn_pipe.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: conv_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: gamma_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: beta_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: gn_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: gn_p.as_entire_binding() },
        ],
    });
    let up_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None, layout: &up_pipe.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: gn_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: up_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: up_p.as_entire_binding() },
        ],
    });
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
        pass.set_pipeline(&conv_pipe); pass.set_bind_group(0, &conv_bg, &[]);
        pass.dispatch_workgroups(((yn + 63) / 64) as u32, 1, 1);
    }
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
        pass.set_pipeline(&gn_pipe); pass.set_bind_group(0, &gn_bg, &[]);
        pass.dispatch_workgroups(gp.n * gp.groups, 1, 1);
    }
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
        pass.set_pipeline(&up_pipe); pass.set_bind_group(0, &up_bg, &[]);
        pass.dispatch_workgroups(((un + 63) / 64) as u32, 1, 1);
    }
    enc.copy_buffer_to_buffer(&up_buf, 0, &readback, 0, (un * 4) as u64);
    queue.submit(Some(enc.finish()));
    let slice = readback.slice(..);
    slice.map_async(wgpu::MapMode::Read, |r| r.expect("map"));
    device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
    let data = slice.get_mapped_range();
    let got: &[f32] = bytemuck::cast_slice(&data);
    let mut max_err = 0.0f32;
    for (a, b) in got.iter().zip(expect.iter()) { max_err = max_err.max((a - b).abs()); }
    println!("vae-mini conv3d->gn+silu->upsample3d out={} max_abs_err={max_err:.3e}", un);
    assert!(max_err < 1e-5, "vae mini mismatch");
    println!("VULKAN VAE MINI OK");
}
