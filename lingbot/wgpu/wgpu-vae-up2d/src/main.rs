use bytemuck::{Pod, Zeroable};
use std::borrow::Cow;
use wgpu::util::DeviceExt;

const CIN: usize = 384;
const COUT: usize = 192;
const D: usize = 2;
const H: usize = 2;
const W: usize = 2;
const OH: usize = 4;
const OW: usize = 4;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct UpP { c: u32, d: u32, h: u32, w: u32, oh: u32, ow: u32, p0: u32, p1: u32 }
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct ConvP { co: u32, ci: u32, d: u32, h: u32, w: u32, pad: u32, p0: u32, p1: u32 }

const UP_WGSL: &str = r#"
struct P { c: u32, d: u32, h: u32, w: u32, oh: u32, ow: u32, p0: u32, p1: u32 };
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
@group(0) @binding(2) var<uniform> p: P;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = p.c * p.d * p.oh * p.ow;
    if (idx >= total) { return; }
    var t = idx;
    let ow = t % p.ow; t = t / p.ow;
    let oh = t % p.oh; t = t / p.oh;
    let od = t % p.d; t = t / p.d;
    let c = t;
    let ih = oh / 2u; let iw = ow / 2u;
    y[idx] = x[((c * p.d + od) * p.h + ih) * p.w + iw];
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
    let total = p.co * p.d * p.h * p.w;
    if (idx >= total) { return; }
    var t = idx;
    let ow = t % p.w; t = t / p.w;
    let oh = t % p.h; t = t / p.h;
    let od = t % p.d; t = t / p.d;
    let co = t;
    var acc = b[co];
    for (var ci = 0u; ci < p.ci; ci = ci + 1u) {
        for (var kd = 0u; kd < 3u; kd = kd + 1u) {
            let id = i32(od) + i32(kd) - 1;
            if (id < 0 || id >= i32(p.d)) { continue; }
            for (var kh = 0u; kh < 3u; kh = kh + 1u) {
                let ih = i32(oh) + i32(kh) - 1;
                if (ih < 0 || ih >= i32(p.h)) { continue; }
                for (var kw = 0u; kw < 3u; kw = kw + 1u) {
                    let iw = i32(ow) + i32(kw) - 1;
                    if (iw < 0 || iw >= i32(p.w)) { continue; }
                    acc = acc + x[((ci * p.d + u32(id)) * p.h + u32(ih)) * p.w + u32(iw)] * w[((co * p.ci + ci) * 3u + kd) * 9u + kh * 3u + kw];
                }
            }
        }
    }
    y[idx] = acc;
}
"#;

fn up(x: &[f32]) -> Vec<f32> {
    let mut y = vec![0.0f32; CIN * D * OH * OW];
    for c in 0..CIN { for d in 0..D { for h in 0..OH { for w in 0..OW {
        y[((c * D + d) * OH + h) * OW + w] = x[((c * D + d) * H + h / 2) * W + w / 2];
    }}}}
    y
}
fn conv(x: &[f32], w: &[f32], b: &[f32]) -> Vec<f32> {
    let mut y = vec![0.0f32; COUT * D * OH * OW];
    for co in 0..COUT {
        for od in 0..D {
            for oh in 0..OH {
                for ow in 0..OW {
                    let mut acc = b[co];
                    for ci in 0..CIN {
                        for kd in 0..3 {
                            let id = od as isize + kd as isize - 1;
                            if id < 0 || id >= D as isize { continue; }
                            for kh in 0..3 {
                                let ih = oh as isize + kh as isize - 1;
                                if ih < 0 || ih >= OH as isize { continue; }
                                for kw in 0..3 {
                                    let iw = ow as isize + kw as isize - 1;
                                    if iw < 0 || iw >= OW as isize { continue; }
                                    acc += x[((ci * D + id as usize) * OH + ih as usize) * OW + iw as usize]
                                        * w[((co * CIN + ci) * 3 + kd) * 9 + kh * 3 + kw];
                                }
                            }
                        }
                    }
                    y[((co * D + od) * OH + oh) * OW + ow] = acc;
                }
            }
        }
    }
    y
}fn main() {
    let wn = COUT * CIN * 27;
    let mut x = vec![0.0f32; CIN * D * H * W];
    let mut w = vec![0.0f32; wn];
    let mut b = vec![0.0f32; COUT];
    for i in 0..x.len() { x[i] = ((i * 13 % 71) as f32 - 35.0) / 19.0; }
    for i in 0..wn { w[i] = ((i * 17 % 97) as f32 - 48.0) / (CIN as f32 * 8.0); }
    for i in 0..COUT { b[i] = ((i * 7 % 11) as f32 - 5.0) * 0.01; }
    let u = up(&x);
    let expect = conv(&u, &w, &b);
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor { backends: wgpu::Backends::VULKAN, ..Default::default() });
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions { power_preference: wgpu::PowerPreference::HighPerformance, compatible_surface: None, force_fallback_adapter: false })).expect("no adapter");
    let info = adapter.get_info();
    println!("adapter: {} | backend: {:?}", info.name, info.backend);
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("vae-up2d"), required_features: wgpu::Features::empty(), required_limits: wgpu::Limits::downlevel_defaults(), ..Default::default()
    })).expect("device");
    let mk = |d: &[f32]| device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(d), usage: wgpu::BufferUsages::STORAGE });
    let xb = mk(&x); let wb = mk(&w); let bb = mk(&b);
    let ub = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (CIN * D * OH * OW * 4) as u64, usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false });
    let yb = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (COUT * D * OH * OW * 4) as u64, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
    let rb = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (COUT * D * OH * OW * 4) as u64, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
    let upp = UpP { c: CIN as u32, d: D as u32, h: H as u32, w: W as u32, oh: OH as u32, ow: OW as u32, p0: 0, p1: 0 };
    let cp = ConvP { co: COUT as u32, ci: CIN as u32, d: D as u32, h: OH as u32, w: OW as u32, pad: 1, p0: 0, p1: 0 };
    let upb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::bytes_of(&upp), usage: wgpu::BufferUsages::UNIFORM });
    let cpb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::bytes_of(&cp), usage: wgpu::BufferUsages::UNIFORM });
    let um = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: None, source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(UP_WGSL)) });
    let cm = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: None, source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(CONV_WGSL)) });
    let upipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: None, layout: None, module: &um, entry_point: Some("main"), compilation_options: Default::default(), cache: None });
    let cpipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: None, layout: None, module: &cm, entry_point: Some("main"), compilation_options: Default::default(), cache: None });
    let ubg = device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &upipe.get_bind_group_layout(0), entries: &[
        wgpu::BindGroupEntry { binding: 0, resource: xb.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 1, resource: ub.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 2, resource: upb.as_entire_binding() },
    ]});
    let cbg = device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &cpipe.get_bind_group_layout(0), entries: &[
        wgpu::BindGroupEntry { binding: 0, resource: ub.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 1, resource: wb.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 2, resource: bb.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 3, resource: yb.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 4, resource: cpb.as_entire_binding() },
    ]});
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
        p.set_pipeline(&upipe); p.set_bind_group(0, &ubg, &[]); p.dispatch_workgroups(((CIN * D * OH * OW + 63) / 64) as u32, 1, 1);
    }
    {
        let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
        p.set_pipeline(&cpipe); p.set_bind_group(0, &cbg, &[]); p.dispatch_workgroups(((COUT * D * OH * OW + 63) / 64) as u32, 1, 1);
    }
    enc.copy_buffer_to_buffer(&yb, 0, &rb, 0, (COUT * D * OH * OW * 4) as u64);
    queue.submit(Some(enc.finish()));
    let slice = rb.slice(..); slice.map_async(wgpu::MapMode::Read, |r| r.expect("map"));
    device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
    let data = slice.get_mapped_range();
    let got: &[f32] = bytemuck::cast_slice(&data);
    let mut max_err = 0.0f32;
    for (a, b) in got.iter().zip(expect.iter()) { max_err = max_err.max((a - b).abs()); }
    println!("vae-up2d {CIN}->{COUT} D={D} H={OH} W={OW} max_abs_err={max_err:.3e}");
    assert!(max_err < 1e-4, "up2d mismatch");
    println!("VULKAN VAE UP2D OK");
}

