use bytemuck::{Pod, Zeroable};
use std::borrow::Cow;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Params {
    n: u32, cin: u32, cout: u32,
    din: u32, hin: u32, win: u32,
    kd: u32, kh: u32, kw: u32,
    pad: u32, stride: u32,
    dout: u32, hout: u32, wout: u32,
    _p0: u32, _p1: u32,
}

const WGSL: &str = r#"
struct Params {
    n: u32, cin: u32, cout: u32,
    din: u32, hin: u32, win: u32,
    kd: u32, kh: u32, kw: u32,
    pad: u32, stride: u32,
    dout: u32, hout: u32, wout: u32,
    p0: u32, p1: u32,
};
@group(0) @binding(0) var<storage, read> x: array<f32>;       // [N,Cin,D,H,W]
@group(0) @binding(1) var<storage, read> w: array<f32>;       // [Cout,Cin,kD,kH,kW]
@group(0) @binding(2) var<storage, read> b: array<f32>;       // [Cout]
@group(0) @binding(3) var<storage, read_write> y: array<f32>; // [N,Cout,Dout,Hout,Wout]
@group(0) @binding(4) var<uniform> p: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = p.n * p.cout * p.dout * p.hout * p.wout;
    if (idx >= total) { return; }
    var t = idx;
    let ow = t % p.wout; t = t / p.wout;
    let oh = t % p.hout; t = t / p.hout;
    let od = t % p.dout; t = t / p.dout;
    let co = t % p.cout;
    let bn = t / p.cout;
    var acc = b[co];
    for (var ci = 0u; ci < p.cin; ci = ci + 1u) {
        for (var kd = 0u; kd < p.kd; kd = kd + 1u) {
            var id = i32(od * p.stride + kd) - i32(p.pad);
            if (p.p0 != 0u) { id = i32(od * p.stride + kd) - i32(p.kd - 1u); }
            if (id < 0 || id >= i32(p.din)) { continue; }
            for (var kh = 0u; kh < p.kh; kh = kh + 1u) {
                let ih = i32(oh * p.stride + kh) - i32(p.pad);
                if (ih < 0 || ih >= i32(p.hin)) { continue; }
                for (var kw = 0u; kw < p.kw; kw = kw + 1u) {
                    let iw = i32(ow * p.stride + kw) - i32(p.pad);
                    if (iw < 0 || iw >= i32(p.win)) { continue; }
                    let xidx = (((bn * p.cin + ci) * p.din + u32(id)) * p.hin + u32(ih)) * p.win + u32(iw);
                    let widx = (((co * p.cin + ci) * p.kd + kd) * p.kh + kh) * p.kw + kw;
                    acc = acc + x[xidx] * w[widx];
                }
            }
        }
    }
    y[idx] = acc;
}
"#;

fn cpu_conv(x: &[f32], w: &[f32], b: &[f32], p: &Params) -> Vec<f32> {
    let total = (p.n * p.cout * p.dout * p.hout * p.wout) as usize;
    let mut y = vec![0.0f32; total];
    for idx in 0..total {
        let mut t = idx;
        let ow = t % p.wout as usize; t /= p.wout as usize;
        let oh = t % p.hout as usize; t /= p.hout as usize;
        let od = t % p.dout as usize; t /= p.dout as usize;
        let co = t % p.cout as usize;
        let bn = t / p.cout as usize;
        let mut acc = b[co];
        for ci in 0..p.cin as usize {
            for kd in 0..p.kd as usize {
                let mut id = od as i32 * p.stride as i32 + kd as i32 - p.pad as i32;
                if p._p0 != 0 { id = od as i32 * p.stride as i32 + kd as i32 - (p.kd as i32 - 1); }
                if id < 0 || id >= p.din as i32 { continue; }
                for kh in 0..p.kh as usize {
                    let ih = oh as i32 * p.stride as i32 + kh as i32 - p.pad as i32;
                    if ih < 0 || ih >= p.hin as i32 { continue; }
                    for kw in 0..p.kw as usize {
                        let iw = ow as i32 * p.stride as i32 + kw as i32 - p.pad as i32;
                        if iw < 0 || iw >= p.win as i32 { continue; }
                        let xidx = (((bn * p.cin as usize + ci) * p.din as usize + id as usize) * p.hin as usize + ih as usize) * p.win as usize + iw as usize;
                        let widx = (((co * p.cin as usize + ci) * p.kd as usize + kd) * p.kh as usize + kh) * p.kw as usize + kw;
                        acc += x[xidx] * w[widx];
                    }
                }
            }
        }
        y[idx] = acc;
    }
    y
}

fn main() {
    let p = Params { n: 1, cin: 2, cout: 3, din: 4, hin: 4, win: 4,
                     kd: 3, kh: 3, kw: 3, pad: 2, stride: 1,
                     dout: 4, hout: 4, wout: 4, _p0: 1, _p1: 0 };
    let xn = (p.n * p.cin * p.din * p.hin * p.win) as usize;
    let wn = (p.cout * p.cin * p.kd * p.kh * p.kw) as usize;
    let yn = (p.n * p.cout * p.dout * p.hout * p.wout) as usize;
    let mut x = vec![0.0f32; xn];
    let mut w = vec![0.0f32; wn];
    let mut b = vec![0.0f32; p.cout as usize];
    for i in 0..xn { x[i] = ((i * 13 % 71) as f32 - 35.0) / 19.0; }
    for i in 0..wn { w[i] = ((i * 17 % 53) as f32 - 26.0) / 31.0; }
    for i in 0..b.len() { b[i] = ((i * 7 % 11) as f32 - 5.0) * 0.01; }
    let expect = cpu_conv(&x, &w, &b, &p);

    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor { backends: wgpu::Backends::VULKAN, ..Default::default() });
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance, compatible_surface: None, force_fallback_adapter: false,
    })).expect("no Vulkan adapter");
    let info = adapter.get_info();
    println!("adapter: {} | backend: {:?}", info.name, info.backend);
    assert_eq!(info.backend, wgpu::Backend::Vulkan);

    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("conv3d"), required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::downlevel_defaults(), ..Default::default()
    })).expect("device");

    let mk = |data: &[f32]| device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None, contents: bytemuck::cast_slice(data), usage: wgpu::BufferUsages::STORAGE,
    });
    let x_buf = mk(&x); let w_buf = mk(&w); let b_buf = mk(&b);
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
        label: Some("conv3d"), source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(WGSL)),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("conv3d"), layout: None, module: &module, entry_point: Some("main"),
        compilation_options: Default::default(), cache: None,
    });
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None, layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: x_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: w_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: b_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: y_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: p_buf.as_entire_binding() },
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
    println!("conv3d causal={} N={} Cin={} Cout={} D={} H={} W={} k=3 max_abs_err={max_err:.3e}",
             p._p0, p.n, p.cin, p.cout, p.dout, p.hout, p.wout);
    assert!(max_err < 1e-5, "conv3d mismatch");
    println!("VULKAN CONV3D OK");
}


