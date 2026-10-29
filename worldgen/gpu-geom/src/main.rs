use anyhow::{Context, Result};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    w: u32,
    h: u32,
    pad0: u32,
    pad1: u32,
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let mut pano = String::new();
    let mut depth_path = String::new();
    let mut out = String::new();
    let mut w = 1024usize;
    let mut h = 512usize;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--pano" => { pano = args[i + 1].clone(); i += 2; }
            "--depth" => { depth_path = args[i + 1].clone(); i += 2; }
            "--out" => { out = args[i + 1].clone(); i += 2; }
            "--width" => { w = args[i + 1].parse()?; i += 2; }
            "--height" => { h = args[i + 1].parse()?; i += 2; }
            _ => { i += 1; }
        }
    }
    anyhow::ensure!(!pano.is_empty() && !depth_path.is_empty() && !out.is_empty(), "usage: gpu-geom --pano P --depth D --out PLY --width W --height H");

    let img = image::open(&pano)?.to_rgb8();
    let img = image::imageops::resize(&img, w as u32, h as u32, image::imageops::FilterType::Triangle);
    let mut rgb = vec![0u32; w * h];
    for (i, px) in img.pixels().enumerate() {
        rgb[i] = px[0] as u32 | ((px[1] as u32) << 8) | ((px[2] as u32) << 16);
    }
    let dbytes = std::fs::read(&depth_path)?;
    let depth: Vec<f32> = dbytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
    anyhow::ensure!(depth.len() == w * h, "depth has {} values, expected {}", depth.len(), w * h);

    let n = w * h;
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor { backends: wgpu::Backends::VULKAN, ..Default::default() });
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))?;
    println!("{} {:?}", adapter.get_info().name, adapter.get_info().backend);
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: None,
        required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::downlevel_defaults(),
        ..Default::default()
    }))?;

    let params = Params { w: w as u32, h: h as u32, pad0: 0, pad1: 0 };
    let ub = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let rb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&rgb),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let db = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&depth),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let out_size = (n * 17 * 4) as u64;
    let ob = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: out_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let read = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: out_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(include_str!("geom.wgsl").into()),
    });
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
        ],
    });
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[&bgl], push_constant_ranges: &[] });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&pl),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: ub.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: rb.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: db.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: ob.as_entire_binding() },
        ],
    });
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
        cp.set_pipeline(&pipeline);
        cp.set_bind_group(0, &bg, &[]);
        cp.dispatch_workgroups(((n as u32) + 63) / 64, 1, 1);
    }
    enc.copy_buffer_to_buffer(&ob, 0, &read, 0, out_size);
    queue.submit(Some(enc.finish()));

    let slice = read.slice(..);
    slice.map_async(wgpu::MapMode::Read, |r| r.unwrap());
    device.poll(wgpu::PollType::wait_indefinitely())?;
    let data = slice.get_mapped_range();
    let vals: Vec<f32> = data.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
    drop(data);
    read.unmap();

    let mut f = BufWriter::new(File::create(Path::new(&out))?);
    writeln!(f, "ply")?;
    writeln!(f, "format binary_little_endian 1.0")?;
    writeln!(f, "element vertex {}", n)?;
    for p in ["x", "y", "z", "nx", "ny", "nz"] { writeln!(f, "property float {p}")?; }
    for k in 0..3 { writeln!(f, "property float f_dc_{k}")?; }
    writeln!(f, "property float opacity")?;
    for k in 0..3 { writeln!(f, "property float scale_{k}")?; }
    for k in 0..4 { writeln!(f, "property float rot_{k}")?; }
    writeln!(f, "end_header")?;
    for i in 0..n {
        let row = &vals[i * 17..i * 17 + 17];
        for v in row { f.write_all(&v.to_le_bytes())?; }
    }
    println!("gpu-geom {} x {} -> {}", w, h, out);
    Ok(())
}
