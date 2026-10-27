use bytemuck::{Pod, Zeroable};
use std::borrow::Cow;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Params {
    heads: u32,
    new_len: u32,
    cache_len: u32,
    dim: u32,
    scale: f32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

const WGSL: &str = r#"
struct Params { heads: u32, new_len: u32, cache_len: u32, dim: u32, scale: f32, pad0: u32, pad1: u32, pad2: u32 };
@group(0) @binding(0) var<storage, read> q_new: array<f32>;     // [new_len, heads, dim]
@group(0) @binding(1) var<storage, read> k_cache: array<f32>;   // [capacity, heads, dim]
@group(0) @binding(2) var<storage, read> v_cache: array<f32>;   // [capacity, heads, dim]
@group(0) @binding(3) var<storage, read> k_new: array<f32>;     // [new_len, heads, dim]
@group(0) @binding(4) var<storage, read> v_new: array<f32>;     // [new_len, heads, dim]
@group(0) @binding(5) var<storage, read_write> out: array<f32>; // [new_len, heads, dim]
@group(0) @binding(6) var<uniform> p: Params;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let h = gid.x / p.new_len;
    let i = gid.x % p.new_len;
    if (h >= p.heads) { return; }
    let total = p.cache_len + p.new_len;
    var scores: array<f32, 64>;
    var maxs = -1e30;
    let qbase = (i * p.heads + h) * p.dim;

    for (var j = 0u; j < total; j = j + 1u) {
        if (j > p.cache_len + i) { break; }
        var kb = 0u;
        if (j < p.cache_len) { kb = (j * p.heads + h) * p.dim; }
        else { kb = ((j - p.cache_len) * p.heads + h) * p.dim; }
        var dot = 0.0;
        for (var d = 0u; d < p.dim; d = d + 1u) {
            var kv = 0.0;
            if (j < p.cache_len) { kv = k_cache[kb + d]; }
            else { kv = k_new[kb + d]; }
            dot = dot + q_new[qbase + d] * kv;
        }
        scores[j] = dot * p.scale;
        maxs = max(maxs, scores[j]);
    }

    var sum = 0.0;
    for (var j = 0u; j <= p.cache_len + i; j = j + 1u) { sum = sum + exp(scores[j] - maxs); }

    for (var d = 0u; d < p.dim; d = d + 1u) {
        var acc = 0.0;
        for (var j = 0u; j <= p.cache_len + i; j = j + 1u) {
            var kb = 0u;
            if (j < p.cache_len) { kb = (j * p.heads + h) * p.dim; }
            else { kb = ((j - p.cache_len) * p.heads + h) * p.dim; }
            var vv = 0.0;
            if (j < p.cache_len) { vv = v_cache[kb + d]; }
            else { vv = v_new[kb + d]; }
            acc = acc + (exp(scores[j] - maxs) / sum) * vv;
        }
        out[(i * p.heads + h) * p.dim + d] = acc;
    }
}
"#;

fn cpu_full(q: &[f32], k: &[f32], v: &[f32], tokens: usize, heads: usize, dim: usize, scale: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; tokens * heads * dim];
    for t in 0..tokens {
        for h in 0..heads {
            let qbase = (t * heads + h) * dim;
            let mut scores = vec![0.0f32; t + 1];
            let mut maxs = f32::NEG_INFINITY;
            for j in 0..=t {
                let mut dot = 0.0f32;
                for d in 0..dim { dot += q[qbase + d] * k[(j * heads + h) * dim + d]; }
                scores[j] = dot * scale;
                maxs = maxs.max(scores[j]);
            }
            let sum: f32 = scores.iter().map(|s| (s - maxs).exp()).sum();
            for d in 0..dim {
                let mut acc = 0.0f32;
                for j in 0..=t { acc += ((scores[j] - maxs).exp() / sum) * v[(j * heads + h) * dim + d]; }
                out[qbase + d] = acc;
            }
        }
    }
    out
}

fn main() {
    let (tokens, heads, dim, chunk) = (16usize, 4usize, 32usize, 8usize);
    assert_eq!(tokens, chunk * 2);
    let capacity = tokens;
    let scale = 1.0 / (dim as f32).sqrt();
    let mut q = vec![0.0f32; tokens * heads * dim];
    let mut k = vec![0.0f32; tokens * heads * dim];
    let mut v = vec![0.0f32; tokens * heads * dim];
    for i in 0..q.len() {
        q[i] = ((i * 13 % 71) as f32 - 35.0) / 19.0;
        k[i] = ((i * 23 % 67) as f32 - 33.0) / 17.0;
        v[i] = ((i * 31 % 61) as f32 - 30.0) / 13.0;
    }
    let expect = cpu_full(&q, &k, &v, tokens, heads, dim, scale);

    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN, ..Default::default()
    });
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None, force_fallback_adapter: false,
    })).expect("no Vulkan adapter");
    let info = adapter.get_info();
    println!("adapter: {} | backend: {:?}", info.name, info.backend);
    assert_eq!(info.backend, wgpu::Backend::Vulkan);

    let mut limits = wgpu::Limits::downlevel_defaults();
    limits.max_storage_buffers_per_shader_stage = 8;
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("kv-attn"), required_features: wgpu::Features::empty(),
        required_limits: limits, ..Default::default()
    })).expect("device");

    let cache_bytes = (capacity * heads * dim * 4) as u64;
    let cache_k = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cache_k"), size: cache_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false,
    });
    let cache_v = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cache_v"), size: cache_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false,
    });
    let zero = vec![0.0f32; capacity * heads * dim];
    queue.write_buffer(&cache_k, 0, bytemuck::cast_slice(&zero));
    queue.write_buffer(&cache_v, 0, bytemuck::cast_slice(&zero));

    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("kv-attn"), source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(WGSL)),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("kv-attn"), layout: None, module: &module, entry_point: Some("main"),
        compilation_options: Default::default(), cache: None,
    });

    let mut streamed = Vec::with_capacity(tokens * heads * dim);
    for c in 0..2usize {
        let start = c * chunk;
        let q_chunk = &q[start * heads * dim..(start + chunk) * heads * dim];
        let k_chunk = &k[start * heads * dim..(start + chunk) * heads * dim];
        let v_chunk = &v[start * heads * dim..(start + chunk) * heads * dim];
        let q_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("q"), contents: bytemuck::cast_slice(q_chunk), usage: wgpu::BufferUsages::STORAGE,
        });
        let k_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("k"), contents: bytemuck::cast_slice(k_chunk), usage: wgpu::BufferUsages::STORAGE,
        });
        let v_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("v"), contents: bytemuck::cast_slice(v_chunk), usage: wgpu::BufferUsages::STORAGE,
        });
        let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("out"), size: (chunk * heads * dim * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false,
        });
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"), size: (chunk * heads * dim * 4) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false,
        });
        let params = Params { heads: heads as u32, new_len: chunk as u32, cache_len: (c * chunk) as u32,
                              dim: dim as u32, scale, _pad0: 0, _pad1: 0, _pad2: 0 };
        let p_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("params"), contents: bytemuck::bytes_of(&params), usage: wgpu::BufferUsages::UNIFORM,
        });
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg"), layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: q_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: cache_k.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: cache_v.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: k_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: v_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: out_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: p_buf.as_entire_binding() },
            ],
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("enc") });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("kv-attn"), timestamp_writes: None });
            pass.set_pipeline(&pipeline); pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups((heads * chunk) as u32, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&out_buf, 0, &readback, 0, (chunk * heads * dim * 4) as u64);
        queue.submit(Some(encoder.finish()));
        let slice = readback.slice(..);
        slice.map_async(wgpu::MapMode::Read, |r| r.expect("map"));
        device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
        let data = slice.get_mapped_range();
        let got: &[f32] = bytemuck::cast_slice(&data);
        streamed.extend_from_slice(got);
        drop(data);
        readback.unmap();

        // append this chunk's K/V into the cache
        let offset = (c * chunk * heads * dim * 4) as u64;
        queue.write_buffer(&cache_k, offset, bytemuck::cast_slice(k_chunk));
        queue.write_buffer(&cache_v, offset, bytemuck::cast_slice(v_chunk));
        println!("chunk {c}: cache_len={} new_len={chunk}", c * chunk);
    }

    let mut max_err = 0.0f32;
    for (a, b) in streamed.iter().zip(expect.iter()) { max_err = max_err.max((a - b).abs()); }
    println!("tokens={tokens} heads={heads} dim={dim} chunks=2 max_abs_err={max_err:.3e}");
    assert!(max_err < 1e-5, "KV attention mismatch");
    println!("VULKAN KV ATTENTION OK");
}



