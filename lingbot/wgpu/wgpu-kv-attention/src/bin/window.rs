use bytemuck::{Pod, Zeroable};
use std::borrow::Cow;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Params {
    new_len: u32,
    cache_len: u32,
    dim: u32,
    sink: u32,
    window_start: u32,
    cache_end: u32,
    scale: f32,
    _pad: u32,
}

const WGSL: &str = r#"
struct Params { new_len: u32, cache_len: u32, dim: u32, sink: u32, window_start: u32, cache_end: u32, scale: f32, pad: u32 };
@group(0) @binding(0) var<storage, read> q_new: array<f32>;   // [new_len, dim]
@group(0) @binding(1) var<storage, read> k_cache: array<f32>; // [capacity, dim]
@group(0) @binding(2) var<storage, read> v_cache: array<f32>;
@group(0) @binding(3) var<storage, read> k_new: array<f32>;   // [new_len, dim]
@group(0) @binding(4) var<storage, read> v_new: array<f32>;
@group(0) @binding(5) var<storage, read_write> out: array<f32>;
@group(0) @binding(6) var<uniform> p: Params;

fn slot_for(j: u32) -> u32 {
    if (j < p.sink) { return j; }
    return p.sink + (j - p.window_start);
}

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.new_len) { return; }
    var scores: array<f32, 64>;
    for (var z = 0u; z < 64u; z = z + 1u) { scores[z] = -1e30; }
    var maxs = -1e30;
    let total = p.cache_end + p.new_len;

    for (var j = 0u; j < total; j = j + 1u) {
        if (j > p.cache_end + i) { break; }
        var valid = false;
        var kb = 0u;
        if (j >= p.cache_end) {
            kb = (j - p.cache_end) * p.dim;
            valid = true;
        } else if (j < p.sink) {
            kb = j * p.dim;
            valid = true;
        } else if (j >= p.window_start) {
            let slot = slot_for(j);
            if (slot < p.cache_len) { kb = slot * p.dim; valid = true; }
        }
        if (!valid) { continue; }
        var dot = 0.0;
        for (var d = 0u; d < p.dim; d = d + 1u) {
            var kv = 0.0;
            if (j >= p.cache_end) { kv = k_new[kb + d]; } else { kv = k_cache[kb + d]; }
            dot = dot + q_new[i * p.dim + d] * kv;
        }
        scores[j] = dot * p.scale;
        maxs = max(maxs, scores[j]);
    }

    var sum = 0.0;
    for (var j = 0u; j <= p.cache_end + i; j = j + 1u) { sum = sum + exp(scores[j] - maxs); }
    for (var d = 0u; d < p.dim; d = d + 1u) {
        var acc = 0.0;
        for (var j = 0u; j <= p.cache_end + i; j = j + 1u) {
            var valid = false;
            var kb = 0u;
            if (j >= p.cache_end) { kb = (j - p.cache_end) * p.dim; valid = true; }
            else if (j < p.sink) { kb = j * p.dim; valid = true; }
            else if (j >= p.window_start) {
                let slot = slot_for(j);
                if (slot < p.cache_len) { kb = slot * p.dim; valid = true; }
            }
            if (!valid) { continue; }
            var vv = 0.0;
            if (j >= p.cache_end) { vv = v_new[kb + d]; } else { vv = v_cache[kb + d]; }
            acc = acc + (exp(scores[j] - maxs) / sum) * vv;
        }
        out[i * p.dim + d] = acc;
    }
}
"#;

fn cpu_chunk(q: &[f32], k: &[f32], v: &[f32], cache_end: usize, chunk: usize,
             dim: usize, sink: usize, window_start: usize, scale: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; chunk * dim];
    for i in 0..chunk {
        let t = cache_end + i;
        let mut valid = Vec::new();
        for j in 0..=t { if j < sink || j >= window_start { valid.push(j); } }
        let mut maxs = f32::NEG_INFINITY;
        let mut scores = Vec::with_capacity(valid.len());
        for &j in &valid {
            let mut dot = 0.0f32;
            for d in 0..dim { dot += q[t * dim + d] * k[j * dim + d]; }
            let s = dot * scale;
            scores.push(s); maxs = maxs.max(s);
        }
        let sum: f32 = scores.iter().map(|s| (s - maxs).exp()).sum();
        for d in 0..dim {
            let mut acc = 0.0f32;
            for (n, &j) in valid.iter().enumerate() {
                acc += ((scores[n] - maxs).exp() / sum) * v[j * dim + d];
            }
            out[i * dim + d] = acc;
        }
    }
    out
}

fn main() {
    let (tokens, dim, chunk, sink, window) = (16usize, 8usize, 4usize, 2usize, 4usize);
    let capacity = sink + window;
    let scale = 1.0 / (dim as f32).sqrt();
    let mut q = vec![0.0f32; tokens * dim];
    let mut k = vec![0.0f32; tokens * dim];
    let mut v = vec![0.0f32; tokens * dim];
    for i in 0..q.len() {
        q[i] = ((i * 13 % 71) as f32 - 35.0) / 19.0;
        k[i] = ((i * 23 % 67) as f32 - 33.0) / 17.0;
        v[i] = ((i * 31 % 61) as f32 - 30.0) / 13.0;
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
        label: Some("kv-window"), required_features: wgpu::Features::empty(),
        required_limits: limits, ..Default::default()
    })).expect("device");
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("kv-window"), source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(WGSL)),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("kv-window"), layout: None, module: &module, entry_point: Some("main"),
        compilation_options: Default::default(), cache: None,
    });

    let mut cache_end = 0usize;
    for c in 0..(tokens / chunk) {
        let start = c * chunk;
        let window_start = std::cmp::max(sink, cache_end.saturating_sub(window));
        let indices: Vec<usize> = (0..cache_end).filter(|&j| j < sink || j >= window_start).collect();
        let cache_len = indices.len();
        let mut ck = vec![0.0f32; capacity * dim];
        let mut cv = vec![0.0f32; capacity * dim];
        for (slot, &j) in indices.iter().enumerate() {
            ck[slot * dim..(slot + 1) * dim].copy_from_slice(&k[j * dim..(j + 1) * dim]);
            cv[slot * dim..(slot + 1) * dim].copy_from_slice(&v[j * dim..(j + 1) * dim]);
        }
        let q_chunk = &q[start * dim..(start + chunk) * dim];
        let k_chunk = &k[start * dim..(start + chunk) * dim];
        let v_chunk = &v[start * dim..(start + chunk) * dim];
        let mk = |data: &[f32]| device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None, contents: bytemuck::cast_slice(data), usage: wgpu::BufferUsages::STORAGE,
        });
        let q_buf = mk(q_chunk); let k_buf = mk(k_chunk); let v_buf = mk(v_chunk);
        let ck_buf = mk(&ck); let cv_buf = mk(&cv);
        let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: None, size: (chunk * dim * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false,
        });
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: None, size: (chunk * dim * 4) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false,
        });
        let params = Params { new_len: chunk as u32, cache_len: cache_len as u32, dim: dim as u32,
                              sink: sink as u32, window_start: window_start as u32,
                              cache_end: cache_end as u32, scale, _pad: 0 };
        let p_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None, contents: bytemuck::bytes_of(&params), usage: wgpu::BufferUsages::UNIFORM,
        });
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: q_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: ck_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: cv_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: k_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: v_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: out_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: p_buf.as_entire_binding() },
            ],
        });
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            pass.set_pipeline(&pipeline); pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(chunk as u32, 1, 1);
        }
        enc.copy_buffer_to_buffer(&out_buf, 0, &readback, 0, (chunk * dim * 4) as u64);
        queue.submit(Some(enc.finish()));
        let slice = readback.slice(..);
        slice.map_async(wgpu::MapMode::Read, |r| r.expect("map"));
        device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
        let data = slice.get_mapped_range();
        let got: &[f32] = bytemuck::cast_slice(&data);
        let expect = cpu_chunk(&q, &k, &v, cache_end, chunk, dim, sink, window_start, scale);
        let mut err = 0.0f32;
        for (a, b) in got.iter().zip(expect.iter()) { err = err.max((a - b).abs()); }
        println!("chunk {c}: cache_end={cache_end} window_start={window_start} cache_len={cache_len} max_abs_err={err:.3e}");
        assert!(err < 1e-5, "window mismatch");
        drop(data);
        readback.unmap();
        cache_end += chunk;
    }
    println!("tokens={tokens} sink={sink} window={window} VULKAN KV WINDOW OK");
}


