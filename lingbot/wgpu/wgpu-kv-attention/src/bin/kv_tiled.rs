use bytemuck::{Pod, Zeroable};
use half::f16;
use std::borrow::Cow;
use wgpu::util::DeviceExt;

const H: usize = 12;
const D: usize = 128;
const S: usize = 24 * 1560;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Params { h: u32, s: u32, d: u32, scale: f32 }

const SCORE_WGSL: &str = r#"
enable f16;
struct Params { h: u32, s: u32, d: u32, scale: f32 };
@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> k: array<f16>;
@group(0) @binding(2) var<storage, read_write> scores: array<f32>;
@group(0) @binding(3) var<uniform> p: Params;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let g = gid.x;
    let h = g / p.s; let j = g % p.s;
    var dot = 0.0;
    for (var d = 0u; d < p.d; d = d + 1u) {
        dot = dot + q[h * p.d + d] * f32(k[(h * p.s + j) * p.d + d]);
    }
    scores[g] = dot * p.scale;
}
"#;
const SOFTMAX_WGSL: &str = r#"
struct Params { h: u32, s: u32, d: u32, scale: f32 };
@group(0) @binding(0) var<storage, read> scores: array<f32>;
@group(0) @binding(1) var<storage, read_write> probs: array<f32>;
@group(0) @binding(2) var<uniform> p: Params;
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let h = gid.x;
    var maxs = -1e30;
    for (var j = 0u; j < p.s; j = j + 1u) { maxs = max(maxs, scores[h * p.s + j]); }
    var sum = 0.0;
    for (var j = 0u; j < p.s; j = j + 1u) { sum = sum + exp(scores[h * p.s + j] - maxs); }
    for (var j = 0u; j < p.s; j = j + 1u) { probs[h * p.s + j] = exp(scores[h * p.s + j] - maxs) / sum; }
}
"#;
const PV_WGSL: &str = r#"
enable f16;
struct Params { h: u32, s: u32, d: u32, scale: f32 };
@group(0) @binding(0) var<storage, read> probs: array<f32>;
@group(0) @binding(1) var<storage, read> v: array<f16>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> p: Params;
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let g = gid.x;
    let h = g / p.d; let d = g % p.d;
    var acc = 0.0;
    for (var j = 0u; j < p.s; j = j + 1u) {
        acc = acc + probs[h * p.s + j] * f32(v[(h * p.s + j) * p.d + d]);
    }
    out[g] = acc;
}
"#;
fn cpu(q: &[f32], k: &[f16], v: &[f16], scale: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; H * D];
    for h in 0..H {
        let mut maxs = f32::NEG_INFINITY;
        for j in 0..S {
            let mut dot = 0.0f32;
            for d in 0..D { dot += q[h * D + d] * k[(h * S + j) * D + d].to_f32(); }
            maxs = maxs.max(dot * scale);
        }
        let mut sum = 0.0f32;
        for j in 0..S {
            let mut dot = 0.0f32;
            for d in 0..D { dot += q[h * D + d] * k[(h * S + j) * D + d].to_f32(); }
            sum += (dot * scale - maxs).exp();
        }
        for d in 0..D {
            let mut acc = 0.0f32;
            for j in 0..S {
                let mut dot = 0.0f32;
                for dd in 0..D { dot += q[h * D + dd] * k[(h * S + j) * D + dd].to_f32(); }
                acc += ((dot * scale - maxs).exp() / sum) * v[(h * S + j) * D + d].to_f32();
            }
            out[h * D + d] = acc;
        }
    }
    out
}

fn main() {
    println!("allocating S={S} tokens...");
    let mut q = vec![0.0f32; H * D];
    let mut k = vec![f16::from_f32(0.0); H * S * D];
    let mut v = vec![f16::from_f32(0.0); H * S * D];
    let scale = 1.0 / (D as f32).sqrt();
    for i in 0..q.len() { q[i] = ((i * 13 % 71) as f32 - 35.0) / 19.0; }
    for i in 0..k.len() {
        k[i] = f16::from_f32(((i * 17 % 97) as f32 - 48.0) / 37.0);
        v[i] = f16::from_f32(((i * 23 % 89) as f32 - 44.0) / 41.0);
    }
    let sink = 6 * 1560;
    let local = 18 * 1560;
    assert_eq!(sink + local, S);
    let newn = 1560 * D;
    let mut newk = vec![f16::from_f32(0.0); H * newn];
    let mut newv = vec![f16::from_f32(0.0); H * newn];
    for i in 0..H * newn {
        newk[i] = f16::from_f32(((i * 31 % 73) as f32 - 36.0) / 43.0);
        newv[i] = f16::from_f32(((i * 37 % 79) as f32 - 39.0) / 47.0);
    }
    let mut k2 = vec![f16::from_f32(0.0); H * S * D];
    let mut v2 = vec![f16::from_f32(0.0); H * S * D];
    for h in 0..H {
        let src = h * S * D;
        let dst = h * S * D;
        k2[dst..dst + sink * D].copy_from_slice(&k[src..src + sink * D]);
        v2[dst..dst + sink * D].copy_from_slice(&v[src..src + sink * D]);
        let local_src = src + (sink + 1560) * D;
        let local_n = (local - 1560) * D;
        k2[dst + sink * D..dst + sink * D + local_n].copy_from_slice(&k[local_src..local_src + local_n]);
        v2[dst + sink * D..dst + sink * D + local_n].copy_from_slice(&v[local_src..local_src + local_n]);
        let tail = dst + (sink + local - 1560) * D;
        k2[tail..tail + newn].copy_from_slice(&newk[h * newn..(h + 1) * newn]);
        v2[tail..tail + newn].copy_from_slice(&newv[h * newn..(h + 1) * newn]);
    }
    k = k2; v = v2;
    println!("kv evict: sink={sink} local={local} append=1560");
    let expect = cpu(&q, &k, &v, scale);

    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor { backends: wgpu::Backends::VULKAN, ..Default::default() });
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance, compatible_surface: None, force_fallback_adapter: false,
    })).expect("no Vulkan adapter");
    let info = adapter.get_info();
    println!("adapter: {} | backend: {:?}", info.name, info.backend);
    let mut limits = wgpu::Limits::downlevel_defaults();
    limits.max_storage_buffer_binding_size = 128 << 20;
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("kv-tiled"), required_features: wgpu::Features::SHADER_F16,
        required_limits: limits, ..Default::default()
    })).expect("device");
    let q_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(&q), usage: wgpu::BufferUsages::STORAGE });
    let k_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(&k), usage: wgpu::BufferUsages::STORAGE });
    let v_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(&v), usage: wgpu::BufferUsages::STORAGE });
    let scores = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (H * S * 4) as u64, usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false });
    let probs = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (H * S * 4) as u64, usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false });
    let o_buf = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (H * D * 4) as u64, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
    let rb = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (H * D * 4) as u64, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
    let params = Params { h: H as u32, s: S as u32, d: D as u32, scale };
    let pb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::bytes_of(&params), usage: wgpu::BufferUsages::UNIFORM });
    let mk = |src: &str| device.create_shader_module(wgpu::ShaderModuleDescriptor { label: None, source: wgpu::ShaderSource::Wgsl(Cow::Owned(src.to_string())) });
    let sm = mk(SCORE_WGSL); let mm = mk(SOFTMAX_WGSL); let pm = mk(PV_WGSL);
    let score_pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: None, layout: None, module: &sm, entry_point: Some("main"), compilation_options: Default::default(), cache: None });
    let soft_pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: None, layout: None, module: &mm, entry_point: Some("main"), compilation_options: Default::default(), cache: None });
    let pv_pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: None, layout: None, module: &pm, entry_point: Some("main"), compilation_options: Default::default(), cache: None });
    let score_bg = device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &score_pipe.get_bind_group_layout(0), entries: &[
        wgpu::BindGroupEntry { binding: 0, resource: q_buf.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 1, resource: k_buf.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 2, resource: scores.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 3, resource: pb.as_entire_binding() },
    ]});
    let soft_bg = device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &soft_pipe.get_bind_group_layout(0), entries: &[
        wgpu::BindGroupEntry { binding: 0, resource: scores.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 1, resource: probs.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 2, resource: pb.as_entire_binding() },
    ]});
    let pv_bg = device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &pv_pipe.get_bind_group_layout(0), entries: &[
        wgpu::BindGroupEntry { binding: 0, resource: probs.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 1, resource: v_buf.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 2, resource: o_buf.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 3, resource: pb.as_entire_binding() },
    ]});
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
        p.set_pipeline(&score_pipe); p.set_bind_group(0, &score_bg, &[]); p.dispatch_workgroups(((H * S + 63) / 64) as u32, 1, 1);
    }
    {
        let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
        p.set_pipeline(&soft_pipe); p.set_bind_group(0, &soft_bg, &[]); p.dispatch_workgroups(H as u32, 1, 1);
    }
    {
        let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
        p.set_pipeline(&pv_pipe); p.set_bind_group(0, &pv_bg, &[]); p.dispatch_workgroups((H * D) as u32, 1, 1);
    }
    enc.copy_buffer_to_buffer(&o_buf, 0, &rb, 0, (H * D * 4) as u64);
    let t0 = std::time::Instant::now();
    queue.submit(Some(enc.finish()));
    let slice = rb.slice(..); slice.map_async(wgpu::MapMode::Read, |r| r.expect("map"));
    device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
    let data = slice.get_mapped_range();
    let got: &[f32] = bytemuck::cast_slice(&data);
    let mut max_err = 0.0f32;
    for (a, b) in got.iter().zip(expect.iter()) { max_err = max_err.max((a - b).abs()); }
    println!("kv-tiled H={H} S={S} D={D} gpu_ms={:.3} max_abs_err={max_err:.3e}", t0.elapsed().as_secs_f64() * 1000.0);
    assert!(max_err < 5e-3, "kv mismatch");
    println!("VULKAN TILED KV OK");
}


