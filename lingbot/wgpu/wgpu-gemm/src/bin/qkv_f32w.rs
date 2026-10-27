use bytemuck::{Pod, Zeroable};
use half::f16;
use std::borrow::Cow;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Params { m: u32, n: u32, k: u32, _pad: u32 }

const WGSL: &str = r#"
enable f16;
struct Params { m: u32, n: u32, k: u32, pad: u32 };
@group(0) @binding(0) var<storage, read> a: array<f32>;   // [M, K]
@group(0) @binding(1) var<storage, read> b: array<f32>;   // [N, K]
@group(0) @binding(2) var<storage, read_write> c: array<f32>; // [M, N]
@group(0) @binding(3) var<uniform> p: Params;

var<workgroup> As: array<f32, 256>;
var<workgroup> Bs: array<f32, 256>;

@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let tx = lid.x;
    let ty = lid.y;
    let row0 = wid.x * 16u;
    let col0 = wid.y * 16u;
    let row = row0 + ty;
    let col = col0 + tx;
    var acc = 0.0;

    var k0 = 0u;
    loop {
        if (k0 >= p.k) { break; }
        if (row < p.m && (k0 + tx) < p.k) { As[ty * 16u + tx] = a[row * p.k + k0 + tx]; }
        else { As[ty * 16u + tx] = 0.0; }
        // B tile is indexed by the *local y* row, not by the output column.
        if ((col0 + ty) < p.n && (k0 + tx) < p.k) { Bs[ty * 16u + tx] = b[(col0 + ty) * p.k + k0 + tx]; }
        else { Bs[ty * 16u + tx] = 0.0; }
        workgroupBarrier();

        var kk = 0u;
        loop {
            if (kk >= 16u) { break; }
            acc = acc + As[ty * 16u + kk] * Bs[tx * 16u + kk];
            kk = kk + 1u;
        }
        workgroupBarrier();
        k0 = k0 + 16u;
    }

    if (row < p.m && col < p.n) { c[row * p.n + col] = acc; }
}
"#;

fn cpu_gemm(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for kk in 0..k { acc += a[i * k + kk] * b[j * k + kk]; }
            c[i * n + j] = acc;
        }
    }
    c
}

fn main() {
    let seed: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(0);
    let out_path = std::env::args().nth(2).unwrap_or_else(|| "q.bin".to_string());
    let m: usize = std::env::var("QKV_M").ok().and_then(|s| s.parse().ok()).unwrap_or(1560);
    let n: usize = std::env::var("QKV_N").ok().and_then(|s| s.parse().ok()).unwrap_or(1536);
    let k: usize = std::env::var("QKV_K").ok().and_then(|s| s.parse().ok()).unwrap_or(1536);
    let in_path = std::env::var("QKV_INPUT_FILE").unwrap_or_else(|_| "D:/3DWord/itseez3d/ai-lab/lingbot-world/port/wgpu-qkv-rope-h0.bin".to_string());
    let rawa = std::fs::read(in_path).expect("input");
    let a16: Vec<f32> = bytemuck::cast_slice::<u8, f32>(&rawa)[..m * k].to_vec();
    let bq: Vec<f32> = if let Ok(path) = std::env::var("QKV_WEIGHT_FILE") {
        let raw = std::fs::read(path).expect("weight file");
        bytemuck::cast_slice::<u8, f32>(&raw)[..n * k].to_vec()
    } else {
        let mut b = vec![0.0f32; n * k];
        for i in 0..b.len() { b[i] = (((i * 29 + seed * 101) % 89) as f32 - 44.0) / 23.0; }
        b
    };
    let bias: Vec<f32> = if let Ok(path) = std::env::var("QKV_BIAS_FILE") {
        let raw = std::fs::read(path).expect("bias file");
        bytemuck::cast_slice::<u8, f32>(&raw)[..n].to_vec()
    } else { vec![0.0; n] };
    let aq: Vec<f32> = a16.clone();
    let mut expect = vec![0.0f32; n];
    for j in 0..n {
        let mut acc = 0.0f32;
        for kk in 0..k { acc += aq[kk] * bq[j * k + kk]; }
        expect[j] = acc + bias[j];
    }

    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN,
        ..Default::default()
    });
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    })).expect("no Vulkan adapter");
    let info = adapter.get_info();
    println!("adapter: {} | backend: {:?} | type: {:?}", info.name, info.backend, info.device_type);
    assert_eq!(info.backend, wgpu::Backend::Vulkan);

    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("gemm"),
        required_features: wgpu::Features::SHADER_F16,
        required_limits: wgpu::Limits::downlevel_defaults(),
        ..Default::default()
    })).expect("device");

    let a_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("a"), contents: bytemuck::cast_slice(&a16), usage: wgpu::BufferUsages::STORAGE,
    });
    let b_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("b"), contents: bytemuck::cast_slice(&bq), usage: wgpu::BufferUsages::STORAGE,
    });
    let c_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("c"), size: (m * n * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let params = Params { m: m as u32, n: n as u32, k: k as u32, _pad: 0 };
    let p_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("params"), contents: bytemuck::bytes_of(&params), usage: wgpu::BufferUsages::UNIFORM,
    });
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"), size: (m * n * 4) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("gemm"), source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(WGSL)),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("gemm"), layout: None, module: &module, entry_point: Some("main"),
        compilation_options: Default::default(), cache: None,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bg"), layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: a_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: b_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: c_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: p_buf.as_entire_binding() },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("enc") });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("gemm"), timestamp_writes: None });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(((m + 15) / 16) as u32, ((n + 15) / 16) as u32, 1);
    }
    encoder.copy_buffer_to_buffer(&c_buf, 0, &readback, 0, (m * n * 4) as u64);
    let t0 = std::time::Instant::now();
    queue.submit(Some(encoder.finish()));

    let slice = readback.slice(..);
    slice.map_async(wgpu::MapMode::Read, |r| r.expect("map"));
    device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
    println!("gpu_ms={:.3}", t0.elapsed().as_secs_f64() * 1000.0);
    let data = slice.get_mapped_range();
    let got: &[f32] = bytemuck::cast_slice(&data);
    let mut max_err = 0.0f32;
    for j in 0..n { max_err = max_err.max((got[j] + bias[j] - expect[j]).abs()); }
    let mut outf32 = got.to_vec();
    for t in 0..m { for j in 0..n { let i=t*n+j; outf32[i] += bias[j]; } }
    std::fs::write(out_path, bytemuck::cast_slice(&outf32)).expect("write f32");
    println!("M={m} N={n} K={k} max_abs_err={max_err:.3e}");
    assert!(max_err < 5e-2, "QKV-f32w mismatch");
    println!("VULKAN GEMM F32W OK");
}





























