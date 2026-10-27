use bytemuck::{Pod, Zeroable};
use std::borrow::Cow;
use wgpu::util::DeviceExt;
use half::f16;

const TT: usize = 1;
const HH: usize = 30;
const WW: usize = 52;
const TOK: usize = TT * HH * WW;
const HEADS: usize = 12;
const HD: usize = 128;
const HALF: usize = HD / 2;
const RT: usize = 22;
const RH: usize = 21;
const RW: usize = 21;

const WGSL: &str = r#"
const TOK: u32 = 1560u;
const HEADS: u32 = 12u;
const HD: u32 = 128u;
const HALF: u32 = 64u;
const RT: u32 = 22u;
const RH: u32 = 21u;
const R_W: u32 = 21u;
@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> k: array<f32>;
@group(0) @binding(2) var<storage, read_write> qo: array<f32>;
@group(0) @binding(3) var<storage, read_write> ko: array<f32>;
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let th = gid.x;
    let tok = th / HEADS;
    let head = th % HEADS;
    if (tok >= TOK) { return; }
    let w = tok % 52u;
    let h = (tok / 52u) % 30u;
    let t = tok / 1560u;
    let base = (tok * HEADS + head) * HD;
    for (var d = 0u; d < HALF; d = d + 1u) {
        var pos = t; var dim = 44u; var local = d;
        if (d >= RT) {
            if (d < RT + RH) { pos = h; dim = 42u; local = d - RT; }
            else { pos = w; dim = 42u; local = d - RT - RH; }
        }
        let freq = 1.0 / pow(10000.0, 2.0 * f32(d) / 128.0);
        let ang = f32(pos) * freq;
        let c = cos(ang); let s = sin(ang);
        let idx = base + 2u * d;
        let q0 = q[idx]; let q1 = q[idx + 1u];
        qo[idx] = q0 * c - q1 * s;
        qo[idx + 1u] = q0 * s + q1 * c;
        let k0 = k[idx]; let k1 = k[idx + 1u];
        ko[idx] = k0 * c - k1 * s;
        ko[idx + 1u] = k0 * s + k1 * c;
    }
}
"#;

fn cpu_rope(x: &[f32]) -> Vec<f32> {
    let mut y = vec![0.0f32; x.len()];
    for tok in 0..TOK {
        let w = tok % 52; let h = (tok / 52) % 30; let t = tok / 1560;
        for head in 0..HEADS {
            let base = (tok * HEADS + head) * HD;
            for d in 0..HALF {
                let (pos, dim, local) = if d < RT { (t, 44, d) }
                    else if d < RT + RH { (h, 42, d - RT) }
                    else { (w, 42, d - RT - RH) };
                let freq = 1.0f32 / 10000.0f32.powf(2.0 * d as f32 / 128.0);
                let ang = pos as f32 * freq;
                let c = ang.cos(); let s = ang.sin();
                let i = base + 2 * d;
                y[i] = x[i] * c - x[i + 1] * s;
                y[i + 1] = x[i] * s + x[i + 1] * c;
            }
        }
    }
    y
}

fn main() {
    let n = TOK * HEADS * HD;
    let rawq = std::fs::read("D:/3DWord/itseez3d/ai-lab/lingbot-world/port/wgpu-qkv-rope-q.bin").expect("q.bin");
    let rawk = std::fs::read("D:/3DWord/itseez3d/ai-lab/lingbot-world/port/wgpu-qkv-rope-k.bin").expect("k.bin");
    let q16: &[f16] = bytemuck::cast_slice(&rawq);
    let k16: &[f16] = bytemuck::cast_slice(&rawk);
    let n_expected = TOK * HEADS * HD;
    assert!(q16.len() >= n_expected, "q.bin too small: {}", q16.len());
    assert!(k16.len() >= n_expected, "k.bin too small: {}", k16.len());
    let mut q = vec![0.0f32; n];
    let mut k = vec![0.0f32; n];
    for i in 0..n { q[i] = q16[i].to_f32(); k[i] = k16[i].to_f32(); }
    let eq = cpu_rope(&q);
    let ek = cpu_rope(&k);

    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor { backends: wgpu::Backends::VULKAN, ..Default::default() });
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance, compatible_surface: None, force_fallback_adapter: false,
    })).expect("no Vulkan adapter");
    let info = adapter.get_info();
    println!("adapter: {} | backend: {:?}", info.name, info.backend);
    assert_eq!(info.backend, wgpu::Backend::Vulkan);
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("rope3d"), required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::downlevel_defaults(), ..Default::default()
    })).expect("device");
    let mk = |data: &[f32]| device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None, contents: bytemuck::cast_slice(data), usage: wgpu::BufferUsages::STORAGE,
    });
    let q_buf = mk(&q); let k_buf = mk(&k);
    let qo_buf = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (n * 4) as u64, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
    let ko_buf = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (n * 4) as u64, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
    let qr = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (n * 4) as u64, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
    let kr = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (n * 4) as u64, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: None, source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(WGSL)) });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: None, layout: None, module: &module, entry_point: Some("main"), compilation_options: Default::default(), cache: None });
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &pipeline.get_bind_group_layout(0), entries: &[
        wgpu::BindGroupEntry { binding: 0, resource: q_buf.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 1, resource: k_buf.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 2, resource: qo_buf.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 3, resource: ko_buf.as_entire_binding() },
    ]});
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
        p.set_pipeline(&pipeline); p.set_bind_group(0, &bg, &[]); p.dispatch_workgroups((TOK * HEADS) as u32, 1, 1);
    }
    enc.copy_buffer_to_buffer(&qo_buf, 0, &qr, 0, (n * 4) as u64);
    enc.copy_buffer_to_buffer(&ko_buf, 0, &kr, 0, (n * 4) as u64);
    queue.submit(Some(enc.finish()));
    let sq = qr.slice(..); sq.map_async(wgpu::MapMode::Read, |r| r.expect("map"));
    let sk = kr.slice(..); sk.map_async(wgpu::MapMode::Read, |r| r.expect("map"));
    device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
    let dq = sq.get_mapped_range(); let dk = sk.get_mapped_range();
    let gq: &[f32] = bytemuck::cast_slice(&dq);
    let gk: &[f32] = bytemuck::cast_slice(&dk);
    let mut max_err = 0.0f32;
    for tok in [0usize, 1, 51, 52, 53, 1559] { for h in 0..HEADS { for d in 0..HD { let idx = (tok * HEADS + h) * HD + d; max_err = max_err.max((gq[idx] - eq[idx]).abs()); } } }
    for tok in [0usize, 1, 51, 52, 53, 1559] { for h in 0..HEADS { for d in 0..HD { let idx = (tok * HEADS + h) * HD + d; max_err = max_err.max((gk[idx] - ek[idx]).abs()); } } }
    println!("rope3d-q1560 tokens={TOK} heads={HEADS} hd={HD} selected max_abs_err={max_err:.3e}");
    assert!(max_err < 1e-3, "rope3d mismatch");
    std::fs::write("D:/3DWord/itseez3d/ai-lab/lingbot-world/port/wgpu-qkv-rope-qr.bin", bytemuck::cast_slice(gq)).expect("write qr");
    std::fs::write("D:/3DWord/itseez3d/ai-lab/lingbot-world/port/wgpu-qkv-rope-kr.bin", bytemuck::cast_slice(gk)).expect("write kr");
    println!("VULKAN 3D ROPE OK");
}





