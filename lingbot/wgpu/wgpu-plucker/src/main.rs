use bytemuck::{Pod, Zeroable};
use std::borrow::Cow;
use wgpu::util::DeviceExt;

const H: usize = 480;
const W: usize = 832;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Params { c2w: [f32; 16], k: [f32; 4], h: u32, w: u32, p0: u32, p1: u32 }

const WGSL: &str = r#"
struct Params { c2w: array<vec4<f32>, 4>, k: vec4<f32>, h: u32, w: u32, p0: u32, p1: u32 };
@group(0) @binding(0) var<storage, read_write> y: array<f32>;
@group(0) @binding(1) var<uniform> p: Params;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= p.h * p.w) { return; }
    let x = f32(idx % p.w) + 0.5;
    let py = f32(idx / p.w) + 0.5;
    let fx = p.k.x; let fy = p.k.y; let cx = p.k.z; let cy = p.k.w;
    var d = vec3<f32>((x - cx) / fx, (py - cy) / fy, 1.0);
    d = normalize(d);
    let r0 = vec3<f32>(p.c2w[0].x, p.c2w[0].y, p.c2w[0].z);
    let r1 = vec3<f32>(p.c2w[1].x, p.c2w[1].y, p.c2w[1].z);
    let r2 = vec3<f32>(p.c2w[2].x, p.c2w[2].y, p.c2w[2].z);
    let dw = vec3<f32>(dot(r0, d), dot(r1, d), dot(r2, d));
    let o = vec3<f32>(p.c2w[0].w, p.c2w[1].w, p.c2w[2].w);
    y[0u * p.h * p.w + idx] = o.x;
    y[1u * p.h * p.w + idx] = o.y;
    y[2u * p.h * p.w + idx] = o.z;
    y[3u * p.h * p.w + idx] = dw.x;
    y[4u * p.h * p.w + idx] = dw.y;
    y[5u * p.h * p.w + idx] = dw.z;
}
"#;

fn main() {
    let mut c2w = [0.0f32; 16];
    for i in 0..4 { c2w[i * 4 + i] = 1.0; }
    c2w[3] = 0.1; c2w[7] = -0.2; c2w[11] = 0.3;
    let k = [800.0, 800.0, W as f32 / 2.0, H as f32 / 2.0];
    let params = Params { c2w, k, h: H as u32, w: W as u32, p0: 0, p1: 0 };
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor { backends: wgpu::Backends::VULKAN, ..Default::default() });
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions { power_preference: wgpu::PowerPreference::HighPerformance, compatible_surface: None, force_fallback_adapter: false })).expect("no adapter");
    let info = adapter.get_info();
    println!("adapter: {} | backend: {:?}", info.name, info.backend);
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor { label: Some("plucker"), required_features: wgpu::Features::empty(), required_limits: wgpu::Limits::downlevel_defaults(), ..Default::default() })).expect("device");
    let yb = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (6*H*W*4) as u64, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
    let rb = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (6*H*W*4) as u64, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
    let pb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::bytes_of(&params), usage: wgpu::BufferUsages::UNIFORM });
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: None, source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(WGSL)) });
    let pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: None, layout: None, module: &module, entry_point: Some("main"), compilation_options: Default::default(), cache: None });
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &pipe.get_bind_group_layout(0), entries: &[
        wgpu::BindGroupEntry { binding: 0, resource: yb.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 1, resource: pb.as_entire_binding() },
    ]});
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    { let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None }); p.set_pipeline(&pipe); p.set_bind_group(0, &bg, &[]); p.dispatch_workgroups(((H*W+63)/64) as u32, 1, 1); }
    enc.copy_buffer_to_buffer(&yb,0,&rb,0,(6*H*W*4) as u64);
    queue.submit(Some(enc.finish()));
    let slice=rb.slice(..); slice.map_async(wgpu::MapMode::Read,|r|r.expect("map")); device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
    let data=slice.get_mapped_range(); let got:&[f32]=bytemuck::cast_slice(&data);
    let mut max_err=0.0f32;
    for idx in [0usize,1,W-1,W,W*H-1] {
        let x= (idx%W) as f32+0.5; let y=(idx/W) as f32+0.5;
        let mut d=[(x-k[2])/k[0],(y-k[3])/k[1],1.0f32];
        let n=(d[0]*d[0]+d[1]*d[1]+d[2]*d[2]).sqrt(); for v in d.iter_mut(){*v/=n;}
        let dw=[d[0],d[1],d[2]];
        let o=[c2w[3],c2w[7],c2w[11]];
        let exp=[o[0],o[1],o[2],dw[0],dw[1],dw[2]];
        for c in 0..6 { max_err=max_err.max((got[c*H*W+idx]-exp[c]).abs()); }
    }
    std::fs::write("D:/3DWord/itseez3d/ai-lab/lingbot-world/port/wgpu-plucker-480.bin",bytemuck::cast_slice(got)).expect("write");
    println!("plucker H={H} W={W} sampled_max_abs_err={max_err:.3e}");
    assert!(max_err<1e-4,"plucker mismatch");
    println!("VULKAN PLUCKER OK");
}



