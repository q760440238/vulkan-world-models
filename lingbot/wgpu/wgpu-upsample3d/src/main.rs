use bytemuck::{Pod, Zeroable};
use std::borrow::Cow;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Params {
    n: u32, c: u32, din: u32, hin: u32, win: u32,
    scale: u32, dout: u32, hout: u32, wout: u32,
    _p0: u32, _p1: u32, _p2: u32,
}

const WGSL: &str = r#"
struct Params {
    n: u32, c: u32, din: u32, hin: u32, win: u32,
    scale: u32, dout: u32, hout: u32, wout: u32,
    p0: u32, p1: u32, p2: u32,
};
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
@group(0) @binding(2) var<uniform> p: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = p.n * p.c * p.dout * p.hout * p.wout;
    if (idx >= total) { return; }
    var t = idx;
    let ow = t % p.wout; t = t / p.wout;
    let oh = t % p.hout; t = t / p.hout;
    let od = t % p.dout; t = t / p.dout;
    let co = t % p.c;
    let bn = t / p.c;
    let iw = ow / p.scale;
    let ih = oh / p.scale;
    let id = od / p.scale;
    let src = (((bn * p.c + co) * p.din + id) * p.hin + ih) * p.win + iw;
    y[idx] = x[src];
}
"#;

fn cpu_up(x: &[f32], p: &Params) -> Vec<f32> {
    let mut y = vec![0.0f32; (p.n * p.c * p.dout * p.hout * p.wout) as usize];
    for idx in 0..y.len() {
        let mut t = idx;
        let ow = t % p.wout as usize; t /= p.wout as usize;
        let oh = t % p.hout as usize; t /= p.hout as usize;
        let od = t % p.dout as usize; t /= p.dout as usize;
        let co = t % p.c as usize;
        let bn = t / p.c as usize;
        let iw = ow / p.scale as usize;
        let ih = oh / p.scale as usize;
        let id = od / p.scale as usize;
        y[idx] = x[(((bn * p.c as usize + co) * p.din as usize + id) * p.hin as usize + ih) * p.win as usize + iw];
    }
    y
}

fn main() {
    let p = Params { n: 1, c: 2, din: 2, hin: 2, win: 2, scale: 2, dout: 4, hout: 4, wout: 4,
                     _p0: 0, _p1: 0, _p2: 0 };
    let xn = (p.n * p.c * p.din * p.hin * p.win) as usize;
    let yn = (p.n * p.c * p.dout * p.hout * p.wout) as usize;
    let mut x = vec![0.0f32; xn];
    for i in 0..xn { x[i] = ((i * 13 % 71) as f32 - 35.0) / 19.0; }
    let expect = cpu_up(&x, &p);

    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor { backends: wgpu::Backends::VULKAN, ..Default::default() });
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance, compatible_surface: None, force_fallback_adapter: false,
    })).expect("no Vulkan adapter");
    let info = adapter.get_info();
    println!("adapter: {} | backend: {:?}", info.name, info.backend);
    assert_eq!(info.backend, wgpu::Backend::Vulkan);

    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("upsample3d"), required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::downlevel_defaults(), ..Default::default()
    })).expect("device");

    let x_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None, contents: bytemuck::cast_slice(&x), usage: wgpu::BufferUsages::STORAGE,
    });
    let y_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: None, size: (yn * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false,
    });
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: None, size: (yn * 4) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false,
    });
    let p_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None, contents: bytemuck::bytes_of(&p), usage: wgpu::BufferUsages::UNIFORM,
    });
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("upsample3d"), source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(WGSL)),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("upsample3d"), layout: None, module: &module, entry_point: Some("main"),
        compilation_options: Default::default(), cache: None,
    });
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None, layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: x_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: y_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: p_buf.as_entire_binding() },
        ],
    });
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
        pass.set_pipeline(&pipeline); pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(((yn + 63) / 64) as u32, 1, 1);
    }
    enc.copy_buffer_to_buffer(&y_buf, 0, &readback, 0, (yn * 4) as u64);
    queue.submit(Some(enc.finish()));
    let slice = readback.slice(..);
    slice.map_async(wgpu::MapMode::Read, |r| r.expect("map"));
    device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
    let data = slice.get_mapped_range();
    let got: &[f32] = bytemuck::cast_slice(&data);
    let mut max_err = 0.0f32;
    for (a, b) in got.iter().zip(expect.iter()) { max_err = max_err.max((a - b).abs()); }
    println!("upsample3d N={} C={} D={}->{} scale={} max_abs_err={max_err:.3e}",
             p.n, p.c, p.din, p.dout, p.scale);
    assert!(max_err < 1e-6, "upsample3d mismatch");
    println!("VULKAN UPSAMPLE3D OK");
}
