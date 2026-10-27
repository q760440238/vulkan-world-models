use bytemuck::{Pod, Zeroable};
use std::borrow::Cow;
use wgpu::util::DeviceExt;

const N: usize = 1;
const CIN: usize = 2;
const COUT: usize = 2;
const D: usize = 4;
const H: usize = 2;
const W: usize = 2;
const K: usize = 3;
const PAD: usize = 1;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Params {
    n: u32, cin: u32, cout: u32, h: u32, w: u32,
    kd: u32, kh: u32, kw: u32, pad: u32, stride: u32,
    _p0: u32, _p1: u32,
}

const WGSL: &str = r#"
struct P { n: u32, cin: u32, cout: u32, h: u32, w: u32, kd: u32, kh: u32, kw: u32,
           pad: u32, stride: u32, p0: u32, p1: u32 };
@group(0) @binding(0) var<storage, read> xnew: array<f32>;  // [n,cin,h,w]
@group(0) @binding(1) var<storage, read> cache: array<f32>; // [n,cin,2,h,w]
@group(0) @binding(2) var<storage, read> w: array<f32>;     // [cout,cin,kd,kh,kw]
@group(0) @binding(3) var<storage, read> b: array<f32>;     // [cout]
@group(0) @binding(4) var<storage, read_write> y: array<f32>; // [n,cout,h,w]
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
    var acc = b[co];
    for (var ci = 0u; ci < p.cin; ci = ci + 1u) {
        for (var kd = 0u; kd < p.kd; kd = kd + 1u) {
            for (var kh = 0u; kh < p.kh; kh = kh + 1u) {
                let ih = i32(oh) + i32(kh) - i32(p.pad);
                if (ih < 0 || ih >= i32(p.h)) { continue; }
                for (var kw = 0u; kw < p.kw; kw = kw + 1u) {
                    let iw = i32(ow) + i32(kw) - i32(p.pad);
                    if (iw < 0 || iw >= i32(p.w)) { continue; }
                    var xi = 0u;
                    if (kd == 2u) {
                        xi = ((bn * p.cin + ci) * p.h + u32(ih)) * p.w + u32(iw);
                    } else {
                        xi = (((bn * p.cin + ci) * 2u + kd) * p.h + u32(ih)) * p.w + u32(iw);
                    }
                    let wi = (((co * p.cin + ci) * p.kd + kd) * p.kh + kh) * p.kw + kw;
                    var xv = 0.0;
                    if (kd == 2u) { xv = xnew[xi]; } else { xv = cache[xi]; }
                    acc = acc + xv * w[wi];
                }
            }
        }
    }
    y[idx] = acc;
}
"#;

fn full_causal_conv(x: &[f32], w: &[f32], b: &[f32]) -> Vec<f32> {
    let mut y = vec![0.0f32; N * COUT * D * H * W];
    for t in 0..D {
        for co in 0..COUT {
            for oh in 0..H {
                for ow in 0..W {
                    let mut acc = b[co];
                    for ci in 0..CIN {
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
                                    let wi = (((co * CIN + ci) * K + kd) * K + kh) * K + kw;
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

fn main() {
    let p = Params { n: N as u32, cin: CIN as u32, cout: COUT as u32, h: H as u32, w: W as u32,
                     kd: K as u32, kh: K as u32, kw: K as u32, pad: PAD as u32, stride: 1,
                     _p0: 0, _p1: 0 };
    let xn = N * CIN * D * H * W;
    let wn = COUT * CIN * K * K * K;
    let mut x = vec![0.0f32; xn];
    let mut w = vec![0.0f32; wn];
    let mut b = vec![0.0f32; COUT];
    for i in 0..xn { x[i] = ((i * 13 % 71) as f32 - 35.0) / 19.0; }
    for i in 0..wn { w[i] = ((i * 17 % 53) as f32 - 26.0) / 31.0; }
    for i in 0..b.len() { b[i] = ((i * 7 % 11) as f32 - 5.0) * 0.01; }
    let expect = full_causal_conv(&x, &w, &b);

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
        label: Some("stream-conv3d"), required_features: wgpu::Features::empty(),
        required_limits: limits, ..Default::default()
    })).expect("device");
    let mk = |data: &[f32]| device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None, contents: bytemuck::cast_slice(data), usage: wgpu::BufferUsages::STORAGE,
    });
    let w_buf = mk(&w); let b_buf = mk(&b);
    let p_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None, contents: bytemuck::bytes_of(&p), usage: wgpu::BufferUsages::UNIFORM,
    });
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("stream-conv3d"), source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(WGSL)),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("stream-conv3d"), layout: None, module: &module, entry_point: Some("main"),
        compilation_options: Default::default(), cache: None,
    });

    let mut cache = vec![0.0f32; N * CIN * 2 * H * W];
    let mut streamed = Vec::with_capacity(N * COUT * D * H * W);
    for t in 0..D {
        let mut frame = vec![0.0f32; N * CIN * H * W];
        for ci in 0..CIN {
            for s in 0..H * W {
                frame[ci * H * W + s] = x[(ci * D + t) * H * W + s];
            }
        }
        let xn_buf = mk(&frame);
        let cache_buf = mk(&cache);
        let y_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: None, size: (N * COUT * H * W * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false,
        });
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: None, size: (N * COUT * H * W * 4) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false,
        });
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: xn_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: cache_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: w_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: b_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: y_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: p_buf.as_entire_binding() },
            ],
        });
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            pass.set_pipeline(&pipeline); pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(((N * COUT * H * W + 63) / 64) as u32, 1, 1);
        }
        enc.copy_buffer_to_buffer(&y_buf, 0, &readback, 0, (N * COUT * H * W * 4) as u64);
        queue.submit(Some(enc.finish()));
        let slice = readback.slice(..);
        slice.map_async(wgpu::MapMode::Read, |r| r.expect("map"));
        device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
        let data = slice.get_mapped_range();
        let got: &[f32] = bytemuck::cast_slice(&data);
        let mut ferr = 0.0f32;
        for co in 0..COUT {
            for s in 0..H * W {
                let exp = expect[(co * D + t) * H * W + s];
                ferr = ferr.max((got[co * H * W + s] - exp).abs());
            }
        }
        println!("stream frame {t}: out={} frame_err={ferr:.3e}", N * COUT * H * W);
        streamed.extend_from_slice(got);
        drop(data);
        readback.unmap();
        // cache layout [N,CIN,2,H,W]: for each channel, slot0=t-2, slot1=t-1
        let hw = H * W;
        for bn in 0..N {
            for ci in 0..CIN {
                let base = (bn * CIN + ci) * 2 * hw;
                let old1 = cache[base + hw..base + 2 * hw].to_vec();
                cache[base..base + hw].copy_from_slice(&old1);
                cache[base + hw..base + 2 * hw]
                    .copy_from_slice(&frame[(bn * CIN + ci) * hw..(bn * CIN + ci + 1) * hw]);
            }
        }
    }
    // streamed layout [t,cout,h,w]; reference layout [cout,t,h,w]
    let mut reordered = vec![0.0f32; streamed.len()];
    for t in 0..D {
        for co in 0..COUT {
            for s in 0..H * W {
                reordered[((co * D + t) * H * W) + s] = streamed[(t * COUT + co) * H * W + s];
            }
        }
    }
    let mut max_err = 0.0f32;
    for (a, b) in reordered.iter().zip(expect.iter()) { max_err = max_err.max((a - b).abs()); }
    println!("stream-conv3d frames={D} k={K} max_abs_err={max_err:.3e}");
    assert!(max_err < 1e-5, "stream conv mismatch");
    println!("VULKAN STREAM CONV3D OK");
}






