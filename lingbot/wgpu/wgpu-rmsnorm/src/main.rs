use bytemuck::{Pod, Zeroable};
use std::borrow::Cow;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Params {
    rows: u32,
    cols: u32,
    eps: f32,
    _pad: u32,
}

const WGSL: &str = r#"
struct Params { rows: u32, cols: u32, eps: f32, pad: u32 };
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> p: Params;

var<workgroup> part: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = wid.x;
    if (row >= p.rows) { return; }
    let base = row * p.cols;
    var acc = 0.0;
    var i = lid.x;
    loop {
        if (i >= p.cols) { break; }
        let v = x[base + i];
        acc = acc + v * v;
        i = i + 256u;
    }
    part[lid.x] = acc;
    workgroupBarrier();
    var stride = 128u;
    loop {
        if (stride == 0u) { break; }
        if (lid.x < stride) { part[lid.x] = part[lid.x] + part[lid.x + stride]; }
        workgroupBarrier();
        stride = stride >> 1u;
    }
    let inv = inverseSqrt(part[0] / f32(p.cols) + p.eps);
    var j = lid.x;
    loop {
        if (j >= p.cols) { break; }
        y[base + j] = x[base + j] * inv * w[j];
        j = j + 256u;
    }
}
"#;

fn cpu_rmsnorm(x: &[f32], w: &[f32], rows: usize, cols: usize, eps: f32) -> Vec<f32> {
    let mut y = vec![0.0f32; rows * cols];
    for r in 0..rows {
        let base = r * cols;
        let mut ss = 0.0f32;
        for c in 0..cols { ss += x[base + c] * x[base + c]; }
        let inv = (ss / cols as f32 + eps).sqrt().recip();
        for c in 0..cols { y[base + c] = x[base + c] * inv * w[c]; }
    }
    y
}

fn main() {
    let rows = 8usize;
    let cols = 2048usize;
    let eps = 1e-5f32;
    let mut x = vec![0.0f32; rows * cols];
    for (i, v) in x.iter_mut().enumerate() {
        *v = ((i * 37 % 101) as f32 - 50.0) / 17.0;
    }
    let w: Vec<f32> = (0..cols).map(|i| 0.5 + (i % 13) as f32 * 0.03).collect();
    let expect = cpu_rmsnorm(&x, &w, rows, cols, eps);

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
    assert_eq!(info.backend, wgpu::Backend::Vulkan, "not Vulkan backend");

    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("rmsnorm"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults(),
            ..Default::default()
        },
    )).expect("device");

    let x_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("x"), contents: bytemuck::cast_slice(&x), usage: wgpu::BufferUsages::STORAGE,
    });
    let w_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("w"), contents: bytemuck::cast_slice(&w), usage: wgpu::BufferUsages::STORAGE,
    });
    let y_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("y"), size: (rows * cols * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let params = Params { rows: rows as u32, cols: cols as u32, eps, _pad: 0 };
    let p_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("params"), contents: bytemuck::bytes_of(&params), usage: wgpu::BufferUsages::UNIFORM,
    });
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"), size: (rows * cols * 4) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("rmsnorm"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(WGSL)),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("rmsnorm"),
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bg"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: x_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: w_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: y_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: p_buf.as_entire_binding() },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("enc") });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("rmsnorm"), timestamp_writes: None });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(rows as u32, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&y_buf, 0, &readback, 0, (rows * cols * 4) as u64);
    queue.submit(Some(encoder.finish()));

    let slice = readback.slice(..);
    slice.map_async(wgpu::MapMode::Read, |r| r.expect("map"));
    device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
    let data = slice.get_mapped_range();
    let got: &[f32] = bytemuck::cast_slice(&data);
    let mut max_err = 0.0f32;
    for (a, b) in got.iter().zip(expect.iter()) {
        max_err = max_err.max((a - b).abs());
    }
    println!("rows={rows} cols={cols} max_abs_err={max_err:.3e}");
    assert!(max_err < 1e-4, "RMSNorm mismatch");
    println!("VULKAN RMSNORM OK");
}


