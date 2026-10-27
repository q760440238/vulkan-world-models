use bytemuck::{Pod, Zeroable};
use std::borrow::Cow;
use wgpu::util::DeviceExt;

const C: usize = 384;
const D: usize = 2;
const H: usize = 2;
const W: usize = 2;
const S: usize = D * H * W;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct ConvP { n: u32, cin: u32, cout: u32, din: u32, hin: u32, win: u32, kd: u32, kh: u32, kw: u32, pad: u32, stride: u32, dout: u32, hout: u32, wout: u32, p0: u32, p1: u32 }
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct AddP { n: u32, p0: u32, p1: u32, p2: u32 }

const CONV_WGSL: &str = r#"
struct P { n: u32, cin: u32, cout: u32, din: u32, hin: u32, win: u32, kd: u32, kh: u32, kw: u32, pad: u32, stride: u32, dout: u32, hout: u32, wout: u32, p0: u32, p1: u32 };
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
            let id = od * p.stride + kd - p.pad;
            if (id < 0 || id >= p.din) { continue; }
            for (var kh = 0u; kh < p.kh; kh = kh + 1u) {
                let ih = oh * p.stride + kh - p.pad;
                if (ih < 0 || ih >= p.hin) { continue; }
                for (var kw = 0u; kw < p.kw; kw = kw + 1u) {
                    let iw = ow * p.stride + kw - p.pad;
                    if (iw < 0 || iw >= p.win) { continue; }
                    let xv = x[(((bn * p.cin + ci) * p.din + id) * p.hin + ih) * p.win + iw];
                    let wv = w[((co * p.cin + ci) * p.kd + kd) * p.kh * p.kw + kh * p.kw + kw];
                    acc = acc + xv * wv;
                }
            }
        }
    }
    y[idx] = acc;
}
"#;
const RMS_WGSL: &str = r#"
const C: u32 = 384u; const S: u32 = 8u;
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> g: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let s = gid.x;
    if (s >= S) { return; }
    var ss = 0.0;
    for (var c = 0u; c < C; c = c + 1u) { let v = x[c * S + s]; ss = ss + v * v; }
    let inv = sqrt(f32(C)) / sqrt(ss + 1e-12);
    for (var c = 0u; c < C; c = c + 1u) { let v = x[c * S + s] * inv * g[c]; y[c * S + s] = v / (1.0 + exp(-v)); }
}
"#;
const ADD_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> p: vec4<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.x) { return; }
    y[i] = a[i] + b[i];
}
"#;

fn conv(x: &[f32], w: &[f32], b: &[f32], cp: &ConvP) -> Vec<f32> {
    let mut y = vec![0.0f32; (cp.cout * cp.dout * cp.hout * cp.wout) as usize];
    for co in 0..cp.cout as usize {
        for od in 0..cp.dout as usize {
            for oh in 0..cp.hout as usize {
                for ow in 0..cp.wout as usize {
                    let mut acc = b[co];
                    for ci in 0..cp.cin as usize {
                        for kd in 0..cp.kd as usize {
                            let id = od as isize + kd as isize - cp.pad as isize;
                            if id < 0 || id >= cp.din as isize { continue; }
                            for kh in 0..cp.kh as usize {
                                let ih = oh as isize + kh as isize - cp.pad as isize;
                                if ih < 0 || ih >= cp.hin as isize { continue; }
                                for kw in 0..cp.kw as usize {
                                    let iw = ow as isize + kw as isize - cp.pad as isize;
                                    if iw < 0 || iw >= cp.win as isize { continue; }
                                    let xv = x[((ci * cp.din as usize + id as usize) * cp.hin as usize + ih as usize) * cp.win as usize + iw as usize];
                                    let wv = w[((co * cp.cin as usize + ci) * cp.kd as usize + kd) * cp.kh as usize * cp.kw as usize + kh * cp.kw as usize + kw];
                                    acc += xv * wv;
                                }
                            }
                        }
                    }
                    y[((co * cp.dout as usize + od) * cp.hout as usize + oh) * cp.wout as usize + ow] = acc;
                }
            }
        }
    }
    y
}fn rms(x: &[f32], g: &[f32]) -> Vec<f32> {
    let mut y = vec![0.0f32; C * S];
    for s in 0..S {
        let mut ss = 0.0f32;
        for c in 0..C { ss += x[c * S + s] * x[c * S + s]; }
        let inv = (C as f32).sqrt() / (ss + 1e-12).sqrt();
        for c in 0..C { y[c * S + s] = x[c * S + s] * inv * g[c]; }
    }
    y
}
fn silu(x: &[f32]) -> Vec<f32> { x.iter().map(|v| v / (1.0 + (-v).exp())).collect() }

fn main() {
    let cp = ConvP { n: 1, cin: C as u32, cout: C as u32, din: D as u32, hin: H as u32, win: W as u32, kd: 3, kh: 3, kw: 3, pad: 1, stride: 1, dout: D as u32, hout: H as u32, wout: W as u32, p0: 0, p1: 0 };
    let ap = AddP { n: (C * S) as u32, p0: 0, p1: 0, p2: 0 };
    let wn = C * C * 27;
    let mut x = vec![0.0f32; C * S];
    let mut w1 = vec![0.0f32; wn];
    let mut w2 = vec![0.0f32; wn];
    let mut b1 = vec![0.0f32; C];
    let mut b2 = vec![0.0f32; C];
    let mut g1 = vec![0.0f32; C];
    let mut g2 = vec![0.0f32; C];
    for i in 0..x.len() { x[i] = ((i * 13 % 71) as f32 - 35.0) / 19.0; }
    for i in 0..wn {
        w1[i] = ((i * 17 % 53) as f32 - 26.0) / 41.0;
        w2[i] = ((i * 19 % 47) as f32 - 23.0) / 43.0;
    }
    for i in 0..C {
        b1[i] = ((i * 7 % 11) as f32 - 5.0) * 0.01;
        b2[i] = ((i * 5 % 9) as f32 - 4.0) * 0.01;
        g1[i] = 1.0 + ((i * 3 % 7) as f32 - 3.0) * 0.02;
        g2[i] = 1.0 + ((i * 5 % 9) as f32 - 4.0) * 0.02;
    }
    let c1 = conv(&x, &w1, &b1, &cp);
    let n1 = rms(&c1, &g1);
    let s1 = silu(&n1);
    let c2 = conv(&s1, &w2, &b2, &cp);
    let expect: Vec<f32> = x.iter().zip(c2.iter()).map(|(a, b)| a + b).collect();

    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor { backends: wgpu::Backends::VULKAN, ..Default::default() });
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions { power_preference: wgpu::PowerPreference::HighPerformance, compatible_surface: None, force_fallback_adapter: false })).expect("no adapter");
    let info = adapter.get_info();
    println!("adapter: {} | backend: {:?}", info.name, info.backend);
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("vae-block384"), required_features: wgpu::Features::empty(), required_limits: wgpu::Limits::downlevel_defaults(), ..Default::default()
    })).expect("device");
    let mk = |data: &[f32]| device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(data), usage: wgpu::BufferUsages::STORAGE });
    let xb = mk(&x); let w1b = mk(&w1); let b1b = mk(&b1); let w2b = mk(&w2); let b2b = mk(&b2); let g1b = mk(&g1); let g2b = mk(&g2);
    let y1 = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (C * S * 4) as u64, usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false });
    let y2 = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (C * S * 4) as u64, usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false });
    let y3 = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (C * S * 4) as u64, usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false });
    let outb = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (C * S * 4) as u64, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
    let rb = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (C * S * 4) as u64, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
    let cpb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::bytes_of(&cp), usage: wgpu::BufferUsages::UNIFORM });
    let apb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::bytes_of(&ap), usage: wgpu::BufferUsages::UNIFORM });
    let cm = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: None, source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(CONV_WGSL)) });
    let rm = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: None, source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(RMS_WGSL)) });
    let am = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: None, source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(ADD_WGSL)) });
    let cpipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: None, layout: None, module: &cm, entry_point: Some("main"), compilation_options: Default::default(), cache: None });
    let rpipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: None, layout: None, module: &rm, entry_point: Some("main"), compilation_options: Default::default(), cache: None });
    let apipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: None, layout: None, module: &am, entry_point: Some("main"), compilation_options: Default::default(), cache: None });
    let bind = |pipe: &wgpu::ComputePipeline, e: &[wgpu::BindGroupEntry]| device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &pipe.get_bind_group_layout(0), entries: e });
    let c1bg = bind(&cpipe, &[wgpu::BindGroupEntry { binding: 0, resource: xb.as_entire_binding() }, wgpu::BindGroupEntry { binding: 1, resource: w1b.as_entire_binding() }, wgpu::BindGroupEntry { binding: 2, resource: b1b.as_entire_binding() }, wgpu::BindGroupEntry { binding: 3, resource: y1.as_entire_binding() }, wgpu::BindGroupEntry { binding: 4, resource: cpb.as_entire_binding() }]);
    let r1bg = bind(&rpipe, &[wgpu::BindGroupEntry { binding: 0, resource: y1.as_entire_binding() }, wgpu::BindGroupEntry { binding: 1, resource: g1b.as_entire_binding() }, wgpu::BindGroupEntry { binding: 2, resource: y2.as_entire_binding() }]);
    let c2bg = bind(&cpipe, &[wgpu::BindGroupEntry { binding: 0, resource: y2.as_entire_binding() }, wgpu::BindGroupEntry { binding: 1, resource: w2b.as_entire_binding() }, wgpu::BindGroupEntry { binding: 2, resource: b2b.as_entire_binding() }, wgpu::BindGroupEntry { binding: 3, resource: y3.as_entire_binding() }, wgpu::BindGroupEntry { binding: 4, resource: cpb.as_entire_binding() }]);
    let abg = bind(&apipe, &[wgpu::BindGroupEntry { binding: 0, resource: xb.as_entire_binding() }, wgpu::BindGroupEntry { binding: 1, resource: y3.as_entire_binding() }, wgpu::BindGroupEntry { binding: 2, resource: outb.as_entire_binding() }, wgpu::BindGroupEntry { binding: 3, resource: apb.as_entire_binding() }]);
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    let mut pass = |pipe: &wgpu::ComputePipeline, bg: &wgpu::BindGroup, g: u32| { let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None }); p.set_pipeline(pipe); p.set_bind_group(0, bg, &[]); p.dispatch_workgroups(g, 1, 1); };
    pass(&cpipe, &c1bg, ((C * S + 63) / 64) as u32);
    pass(&rpipe, &r1bg, S as u32);
    pass(&cpipe, &c2bg, ((C * S + 63) / 64) as u32);
    pass(&apipe, &abg, ((C * S + 63) / 64) as u32);
    enc.copy_buffer_to_buffer(&outb, 0, &rb, 0, (C * S * 4) as u64);
    queue.submit(Some(enc.finish()));
    let slice = rb.slice(..); slice.map_async(wgpu::MapMode::Read, |r| r.expect("map"));
    device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
    let data = slice.get_mapped_range();
    let got: &[f32] = bytemuck::cast_slice(&data);
    let mut max_err = 0.0f32;
    for (a, b) in got.iter().zip(expect.iter()) { max_err = max_err.max((a - b).abs()); }
    println!("vae-block384 C={C} spatial={S} conv3d->rms+silu->conv3d->residual max_abs_err={max_err:.3e}");
    assert!(max_err < 1e-4, "block384 mismatch");
    println!("VULKAN VAE BLOCK384 OK");
}




