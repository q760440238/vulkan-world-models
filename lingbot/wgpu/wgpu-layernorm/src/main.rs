use bytemuck::{Pod, Zeroable};
use std::borrow::Cow;
use wgpu::util::DeviceExt;

const M: usize = 1560;
const C: usize = 1536;

const WGSL: &str = r#"
const C: u32 = 1536u;
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> g: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = gid.x;
    if (t >= 1560u) { return; }
    var mean = 0.0;
    for (var c = 0u; c < C; c = c + 1u) { mean = mean + x[t * C + c]; }
    mean = mean / f32(C);
    var var_ = 0.0;
    for (var c = 0u; c < C; c = c + 1u) { let d = x[t * C + c] - mean; var_ = var_ + d * d; }
    var_ = var_ / f32(C);
    let inv = inverseSqrt(var_ + 1e-6);
    for (var c = 0u; c < C; c = c + 1u) { y[t * C + c] = (x[t * C + c] - mean) * inv * g[c] + b[c]; }
}
"#;

fn main() {
    let mut x = vec![0.0f32; M * C];
    for i in 0..x.len() { x[i] = ((i * 13 % 71) as f32 - 35.0) / 19.0; }
    let mut g = vec![0.0f32; C]; let mut b = vec![0.0f32; C];
    for c in 0..C { g[c] = 1.0 + ((c*7%11) as f32 - 5.0)*0.01; b[c] = ((c*5%13) as f32 - 6.0)*0.01; }
    if let Ok(path)=std::env::var("LN_INPUT") {
        let raw=std::fs::read(path).expect("ln input");
        let x16: Vec<half::f16>=bytemuck::cast_slice::<u8,half::f16>(&raw)[..M*C].to_vec();
        for i in 0..M*C { x[i]=x16[i].to_f32(); }
    }
    if let Ok(path)=std::env::var("LN_GAMMA") {
        let raw=std::fs::read(path).expect("ln gamma");
        let v: Vec<half::f16>=bytemuck::cast_slice::<u8,half::f16>(&raw)[..C].to_vec();
        for i in 0..C { g[i]=v[i].to_f32(); }
    }
    if let Ok(path)=std::env::var("LN_BETA") {
        let raw=std::fs::read(path).expect("ln beta");
        let v: Vec<half::f16>=bytemuck::cast_slice::<u8,half::f16>(&raw)[..C].to_vec();
        for i in 0..C { b[i]=v[i].to_f32(); }
    }
    let mut expect = vec![0.0f32; 2 * C];
    for t in 0..2 {
        let mean: f32 = (0..C).map(|c| x[t*C+c]).sum::<f32>() / C as f32;
        let var: f32 = (0..C).map(|c| (x[t*C+c]-mean)*(x[t*C+c]-mean)).sum::<f32>() / C as f32;
        let inv = 1.0f32 / (var + 1e-6).sqrt();
        for c in 0..C { expect[t*C+c] = (x[t*C+c]-mean)*inv*g[c]+b[c]; }
    }
    let instance=wgpu::Instance::new(&wgpu::InstanceDescriptor{backends:wgpu::Backends::VULKAN,..Default::default()});
    let adapter=pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions{power_preference:wgpu::PowerPreference::HighPerformance,compatible_surface:None,force_fallback_adapter:false})).expect("no adapter");
    let info=adapter.get_info(); println!("adapter: {} | backend: {:?}",info.name,info.backend);
    let (device,queue)=pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor{label:Some("layernorm"),required_features:wgpu::Features::empty(),required_limits:wgpu::Limits::downlevel_defaults(),..Default::default()})).expect("device");
    let xb=device.create_buffer_init(&wgpu::util::BufferInitDescriptor{label:None,contents:bytemuck::cast_slice(&x),usage:wgpu::BufferUsages::STORAGE});
    let gb=device.create_buffer_init(&wgpu::util::BufferInitDescriptor{label:None,contents:bytemuck::cast_slice(&g),usage:wgpu::BufferUsages::STORAGE});
    let bb=device.create_buffer_init(&wgpu::util::BufferInitDescriptor{label:None,contents:bytemuck::cast_slice(&b),usage:wgpu::BufferUsages::STORAGE});
    let yb=device.create_buffer(&wgpu::BufferDescriptor{label:None,size:(M*C*4) as u64,usage:wgpu::BufferUsages::STORAGE|wgpu::BufferUsages::COPY_SRC,mapped_at_creation:false});
    let rb=device.create_buffer(&wgpu::BufferDescriptor{label:None,size:(M*C*4) as u64,usage:wgpu::BufferUsages::COPY_DST|wgpu::BufferUsages::MAP_READ,mapped_at_creation:false});
    let module=device.create_shader_module(wgpu::ShaderModuleDescriptor{label:None,source:wgpu::ShaderSource::Wgsl(Cow::Borrowed(WGSL))});
    let pipe=device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor{label:None,layout:None,module:&module,entry_point:Some("main"),compilation_options:Default::default(),cache:None});
    let bg=device.create_bind_group(&wgpu::BindGroupDescriptor{label:None,layout:&pipe.get_bind_group_layout(0),entries:&[
        wgpu::BindGroupEntry{binding:0,resource:xb.as_entire_binding()},
        wgpu::BindGroupEntry{binding:1,resource:gb.as_entire_binding()},
        wgpu::BindGroupEntry{binding:2,resource:bb.as_entire_binding()},
        wgpu::BindGroupEntry{binding:3,resource:yb.as_entire_binding()}]});
    let mut enc=device.create_command_encoder(&wgpu::CommandEncoderDescriptor{label:None});
    {let mut p=enc.begin_compute_pass(&wgpu::ComputePassDescriptor{label:None,timestamp_writes:None});p.set_pipeline(&pipe);p.set_bind_group(0,&bg,&[]);p.dispatch_workgroups(M as u32,1,1);}
    enc.copy_buffer_to_buffer(&yb,0,&rb,0,(M*C*4) as u64);
    queue.submit(Some(enc.finish()));
    let slice=rb.slice(..);slice.map_async(wgpu::MapMode::Read,|r|r.expect("map"));device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
    let data=slice.get_mapped_range();let got:&[f32]=bytemuck::cast_slice(&data);
    let mut max_err=0.0f32;for i in 0..2*C{max_err=max_err.max((got[i]-expect[i]).abs());}
    if let Ok(path)=std::env::var("LN_OUT") {
        let y16: Vec<half::f16> = got[..M*C].iter().map(|v| half::f16::from_f32(*v)).collect();
        std::fs::write(path, bytemuck::cast_slice(&y16)).expect("ln out");
    }
    println!("layernorm M={M} C={C} first2_max_abs_err={max_err:.3e}");
    println!("VULKAN LAYERNORM OK");
}



