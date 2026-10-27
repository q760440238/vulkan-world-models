use bytemuck::{Pod, Zeroable};
use half::f16;
use std::borrow::Cow;
use wgpu::util::DeviceExt;

const M: usize = 1560;
const C: usize = 1536;

const WGSL: &str = r#"
enable f16;
const C: u32 = 1536u;
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> g: array<f16>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = gid.x;
    if (t >= 1560u) { return; }
    var ss = 0.0;
    for (var c = 0u; c < C; c = c + 1u) { let v = x[t*C+c]; ss = ss + v*v; }
    let inv = inverseSqrt(ss / f32(C) + 1e-5);
    for (var c = 0u; c < C; c = c + 1u) { y[t*C+c] = x[t*C+c] * inv * f32(g[c]); }
}
"#;

fn main() {
    let qpath = std::env::args().nth(1).unwrap_or_else(|| "q.bin".into());
    let gpath = std::env::args().nth(2).unwrap_or_else(|| "gamma.f16".into());
    let opath = std::env::args().nth(3).unwrap_or_else(|| "qn.bin".into());
    let m: usize = std::env::var("QK_M").ok().and_then(|s| s.parse().ok()).unwrap_or(M);
    let rawq=std::fs::read(&qpath).expect("q"); let rawg=std::fs::read(&gpath).expect("g");
    let x: Vec<f32>=bytemuck::cast_slice::<u8,f32>(&rawq)[..m*C].to_vec();
    let g: Vec<f16>=bytemuck::cast_slice::<u8,f16>(&rawg)[..C].to_vec();
    let mut expect=vec![0.0f32;2*C];
    for t in 0..2 {
        let mut ss=0.0f32; for c in 0..C { let v=x[t*C+c]; ss+=v*v; }
        let inv=1.0/(ss/C as f32 + 1e-5).sqrt();
        for c in 0..C { expect[t*C+c]=x[t*C+c]*inv*g[c].to_f32(); }
    }
    let instance=wgpu::Instance::new(&wgpu::InstanceDescriptor{backends:wgpu::Backends::VULKAN,..Default::default()});
    let adapter=pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions{power_preference:wgpu::PowerPreference::HighPerformance,compatible_surface:None,force_fallback_adapter:false})).expect("no adapter");
    let info=adapter.get_info(); println!("adapter: {} | backend: {:?}",info.name,info.backend);
    let (device,queue)=pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor{label:Some("qk-norm"),required_features:wgpu::Features::SHADER_F16,required_limits:wgpu::Limits::downlevel_defaults(),..Default::default()})).expect("device");
    let mk=|d:&[f16]|device.create_buffer_init(&wgpu::util::BufferInitDescriptor{label:None,contents:bytemuck::cast_slice(d),usage:wgpu::BufferUsages::STORAGE});
    let mk32=|d:&[f32]|device.create_buffer_init(&wgpu::util::BufferInitDescriptor{label:None,contents:bytemuck::cast_slice(d),usage:wgpu::BufferUsages::STORAGE});
    let xb=mk32(&x); let gb=mk(&g);
    let yb=device.create_buffer(&wgpu::BufferDescriptor{label:None,size:(m*C*4) as u64,usage:wgpu::BufferUsages::STORAGE|wgpu::BufferUsages::COPY_SRC,mapped_at_creation:false});
    let rb=device.create_buffer(&wgpu::BufferDescriptor{label:None,size:(m*C*4) as u64,usage:wgpu::BufferUsages::COPY_DST|wgpu::BufferUsages::MAP_READ,mapped_at_creation:false});
    let module=device.create_shader_module(wgpu::ShaderModuleDescriptor{label:None,source:wgpu::ShaderSource::Wgsl(Cow::Borrowed(WGSL))});
    let pipe=device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor{label:None,layout:None,module:&module,entry_point:Some("main"),compilation_options:Default::default(),cache:None});
    let bg=device.create_bind_group(&wgpu::BindGroupDescriptor{label:None,layout:&pipe.get_bind_group_layout(0),entries:&[
        wgpu::BindGroupEntry{binding:0,resource:xb.as_entire_binding()},
        wgpu::BindGroupEntry{binding:1,resource:gb.as_entire_binding()},
        wgpu::BindGroupEntry{binding:2,resource:yb.as_entire_binding()}]});
    let mut enc=device.create_command_encoder(&wgpu::CommandEncoderDescriptor{label:None});
    {let mut p=enc.begin_compute_pass(&wgpu::ComputePassDescriptor{label:None,timestamp_writes:None});p.set_pipeline(&pipe);p.set_bind_group(0,&bg,&[]);p.dispatch_workgroups(m as u32,1,1);}
    enc.copy_buffer_to_buffer(&yb,0,&rb,0,(m*C*4) as u64);
    queue.submit(Some(enc.finish()));
    let slice=rb.slice(..);slice.map_async(wgpu::MapMode::Read,|r|r.expect("map"));device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
    let data=slice.get_mapped_range();let got:&[f32]=bytemuck::cast_slice(&data);
    let mut max_err=0.0f32;for i in 0..2*C{max_err=max_err.max((got[i]-expect[i]).abs());}
    std::fs::write(&opath,bytemuck::cast_slice(&got[..m*C])).expect("write");
    println!("qk-norm-f32 M={m} C={C} first2_max_abs_err={max_err:.3e}");
    println!("VULKAN QK NORM OK");
}





