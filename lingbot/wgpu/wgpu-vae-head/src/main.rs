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
struct ConvP { co: u32, ci: u32, d: u32, h: u32, w: u32, pad: u32, p0: u32, p1: u32 }

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
const CONV_WGSL: &str = r#"
struct P { co: u32, ci: u32, d: u32, h: u32, w: u32, pad: u32, p0: u32, p1: u32 };
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(4) var<uniform> p: P;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= p.co * p.d * p.h * p.w) { return; }
    var t = idx;
    let ow = t % p.w; t = t / p.w;
    let oh = t % p.h; t = t / p.h;
    let od = t % p.d; t = t / p.d;
    let co = t;
    var acc = b[co];
    for (var ci = 0u; ci < p.ci; ci = ci + 1u) {
        for (var kd = 0u; kd < 3u; kd = kd + 1u) {
            let id = i32(od) + i32(kd) - 1; if (id < 0 || id >= i32(p.d)) { continue; }
            for (var kh = 0u; kh < 3u; kh = kh + 1u) {
                let ih = i32(oh) + i32(kh) - 1; if (ih < 0 || ih >= i32(p.h)) { continue; }
                for (var kw = 0u; kw < 3u; kw = kw + 1u) {
                    let iw = i32(ow) + i32(kw) - 1; if (iw < 0 || iw >= i32(p.w)) { continue; }
                    acc = acc + x[((ci * p.d + u32(id)) * p.h + u32(ih)) * p.w + u32(iw)] * w[((co * p.ci + ci) * 3u + kd) * 9u + kh * 3u + kw];
                }
            }
        }
    }
    y[idx] = acc;
}
"#;

fn rms(x: &[f32], g: &[f32]) -> Vec<f32> {
    let mut y = vec![0.0f32; C * S];
    for s in 0..S {
        let mut ss = 0.0f32;
        for c in 0..C { ss += x[c * S + s] * x[c * S + s]; }
        let inv = (C as f32).sqrt() / (ss + 1e-12).sqrt();
        for c in 0..C { let v = x[c * S + s] * inv * g[c]; y[c * S + s] = v / (1.0 + (-v).exp()); }
    }
    y
}
fn conv(x: &[f32], w: &[f32], b: &[f32]) -> Vec<f32> {
    let mut y = vec![0.0f32; 3 * S];
    for co in 0..3 {
        for od in 0..D {
            for oh in 0..H {
                for ow in 0..W {
                    let mut acc = b[co];
                    for ci in 0..C {
                        for kd in 0..3 {
                            let id = od as isize + kd as isize - 1;
                            if id < 0 || id >= D as isize { continue; }
                            for kh in 0..3 {
                                let ih = oh as isize + kh as isize - 1;
                                if ih < 0 || ih >= H as isize { continue; }
                                for kw in 0..3 {
                                    let iw = ow as isize + kw as isize - 1;
                                    if iw < 0 || iw >= W as isize { continue; }
                                    acc += x[((ci * D + id as usize) * H + ih as usize) * W + iw as usize]
                                        * w[((co * C + ci) * 3 + kd) * 9 + kh * 3 + kw];
                                }
                            }
                        }
                    }
                    y[((co * D + od) * H + oh) * W + ow] = acc;
                }
            }
        }
    }
    y
}fn main() {
    let mut x = vec![0.0f32; C * S];
    let mut g = vec![0.0f32; C];
    let mut w = vec![0.0f32; 3 * C * 27];
    let mut b = vec![0.0f32; 3];
    for i in 0..x.len() { x[i] = ((i * 13 % 71) as f32 - 35.0) / 19.0; }
    for i in 0..C { g[i] = 1.0 + ((i * 7 % 11) as f32 - 5.0) * 0.01; }
    for i in 0..w.len() { w[i] = ((i * 17 % 97) as f32 - 48.0) / (C as f32 * 8.0); }
    for i in 0..3 { b[i] = ((i * 7 % 11) as f32 - 5.0) * 0.01; }
    let expect = conv(&rms(&x, &g), &w, &b);
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor { backends: wgpu::Backends::VULKAN, ..Default::default() });
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions { power_preference: wgpu::PowerPreference::HighPerformance, compatible_surface: None, force_fallback_adapter: false })).expect("no adapter");
    let info = adapter.get_info();
    println!("adapter: {} | backend: {:?}", info.name, info.backend);
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("vae-head"), required_features: wgpu::Features::empty(), required_limits: wgpu::Limits::downlevel_defaults(), ..Default::default()
    })).expect("device");
    let mk = |d: &[f32]| device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(d), usage: wgpu::BufferUsages::STORAGE });
    let xb = mk(&x); let gb = mk(&g); let wb = mk(&w); let bb = mk(&b);
    let y1 = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (C*S*4) as u64, usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false });
    let yb = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (3*S*4) as u64, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
    let rb = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (3*S*4) as u64, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
    let cp = ConvP { co: 3, ci: C as u32, d: D as u32, h: H as u32, w: W as u32, pad: 1, p0: 0, p1: 0 };
    let cpb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::bytes_of(&cp), usage: wgpu::BufferUsages::UNIFORM });
    let rm = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: None, source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(RMS_WGSL)) });
    let cm = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: None, source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(CONV_WGSL)) });
    let rp = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: None, layout: None, module: &rm, entry_point: Some("main"), compilation_options: Default::default(), cache: None });
    let cpip = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: None, layout: None, module: &cm, entry_point: Some("main"), compilation_options: Default::default(), cache: None });
    let rbg = device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &rp.get_bind_group_layout(0), entries: &[
        wgpu::BindGroupEntry { binding: 0, resource: xb.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 1, resource: gb.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 2, resource: y1.as_entire_binding() },
    ]});
    let cbg = device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &cpip.get_bind_group_layout(0), entries: &[
        wgpu::BindGroupEntry { binding: 0, resource: y1.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 1, resource: wb.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 2, resource: bb.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 3, resource: yb.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 4, resource: cpb.as_entire_binding() },
    ]});
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
        p.set_pipeline(&rp); p.set_bind_group(0, &rbg, &[]); p.dispatch_workgroups(S as u32, 1, 1);
    }
    {
        let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
        p.set_pipeline(&cpip); p.set_bind_group(0, &cbg, &[]); p.dispatch_workgroups(((3*S + 63) / 64) as u32, 1, 1);
    }
    enc.copy_buffer_to_buffer(&yb, 0, &rb, 0, (3*S*4) as u64);
    queue.submit(Some(enc.finish()));
    let slice = rb.slice(..); slice.map_async(wgpu::MapMode::Read, |r| r.expect("map"));
    device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
    let data = slice.get_mapped_range();
    let got: &[f32] = bytemuck::cast_slice(&data);
    let mut max_err = 0.0f32;
    for (a, b) in got.iter().zip(expect.iter()) { max_err = max_err.max((a - b).abs()); }
    println!("vae-head C={C} out=3x2x2x2 max_abs_err={max_err:.3e}");
    assert!(max_err < 1e-4, "head mismatch");
    println!("VULKAN VAE HEAD OK");
}

