use bytemuck::{Pod, Zeroable};
use half::f16;
use std::borrow::Cow;
use wgpu::util::DeviceExt;

const H: usize = 12;
const Q: usize = 1560;
const S: usize = 26;
const D: usize = 128;

const SCORE_WGSL: &str = r#"
enable f16;
const H: u32=12u; const Q: u32=1560u; const S: u32=26u; const D: u32=128u;
@group(0) @binding(0) var<storage, read> qv: array<f32>;
@group(0) @binding(1) var<storage, read> kv: array<f32>;
@group(0) @binding(2) var<storage, read_write> scores: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let g=gid.x; if(g>=H*Q*S){return;}
    let row=g/S; let j=g%S; let h=row/Q; let i=row%Q;
    var dot=0.0;
    for(var d=0u;d<D;d=d+1u){
        dot=dot+qv[(i*H+h)*D+d]*kv[(j*H+h)*D+d];
    }
    scores[g]=dot/sqrt(f32(D));
}
"#;
const SOFT_WGSL: &str = r#"
const H: u32=12u; const Q: u32=1560u; const S: u32=26u;
@group(0) @binding(0) var<storage, read> scores: array<f32>;
@group(0) @binding(1) var<storage, read_write> probs: array<f32>;
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row=gid.x; if(row>=H*Q){return;}
    var maxs=-1e30;
    for(var j=0u;j<S;j=j+1u){maxs=max(maxs,scores[row*S+j]);}
    var sum=0.0;
    for(var j=0u;j<S;j=j+1u){sum=sum+exp(scores[row*S+j]-maxs);}
    for(var j=0u;j<S;j=j+1u){probs[row*S+j]=exp(scores[row*S+j]-maxs)/sum;}
}
"#;
const PV_WGSL: &str = r#"
enable f16;
const H: u32=12u; const Q: u32=1560u; const S: u32=26u; const D: u32=128u;
@group(0) @binding(0) var<storage, read> probs: array<f32>;
@group(0) @binding(1) var<storage, read> vv: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let g=gid.x; if(g>=H*Q*D){return;}
    let row=g/D; let d=g%D; let h=row/Q; let i=row%Q;
    var acc=0.0;
    for(var j=0u;j<S;j=j+1u){acc=acc+probs[row*S+j]*vv[(j*H+h)*D+d];}
    out[(i*H+h)*D+d]=acc;
}
"#;

fn main(){
    let mut q=vec![0.0f32;Q*H*D]; let mut k=vec![0.0f32;S*H*D]; let mut v=vec![0.0f32;S*H*D];
    if let Ok(path)=std::env::var("CROSS_Q") { let raw=std::fs::read(path).expect("q"); q.copy_from_slice(&bytemuck::cast_slice::<u8,f32>(&raw)[..Q*H*D]); }
    else { for i in 0..q.len(){q[i]=((i*13%71) as f32-35.0)/19.0;} }
    if let Ok(path)=std::env::var("CROSS_K") { let raw=std::fs::read(path).expect("k"); k.copy_from_slice(&bytemuck::cast_slice::<u8,f32>(&raw)[..S*H*D]); }
    else { for i in 0..k.len(){k[i]=((i*17%97) as f32-48.0)/37.0;} }
    if let Ok(path)=std::env::var("CROSS_V") { let raw=std::fs::read(path).expect("v"); v.copy_from_slice(&bytemuck::cast_slice::<u8,f32>(&raw)[..S*H*D]); }
    else { for i in 0..v.len(){v[i]=((i*23%89) as f32-44.0)/41.0;} }
    let mut expect=vec![0.0f32;2*H*D];
    for h in 0..H { for i in 0..2 {
        let mut scores=vec![0.0f32;S]; let mut maxs=f32::NEG_INFINITY;
        for j in 0..S { let mut dot=0.0f32; for d in 0..D { dot+=q[(i*H+h)*D+d]*k[(j*H+h)*D+d]; } scores[j]=dot/(D as f32).sqrt(); maxs=maxs.max(scores[j]); }
        let sum:f32=scores.iter().map(|s|(s-maxs).exp()).sum();
        for d in 0..D { let mut acc=0.0f32; for j in 0..S { acc+=(scores[j]-maxs).exp()/sum*v[(j*H+h)*D+d]; } expect[(i*H+h)*D+d]=acc; }
    }}
    let instance=wgpu::Instance::new(&wgpu::InstanceDescriptor{backends:wgpu::Backends::VULKAN,..Default::default()});
    let adapter=pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions{power_preference:wgpu::PowerPreference::HighPerformance,compatible_surface:None,force_fallback_adapter:false})).expect("no adapter");
    let info=adapter.get_info(); println!("adapter: {} | backend: {:?}",info.name,info.backend);
    let mut limits=wgpu::Limits::downlevel_defaults(); limits.max_storage_buffer_binding_size=128<<20;
    let (device,queue)=pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor{label:Some("cross512"),required_features:wgpu::Features::SHADER_F16,required_limits:limits,..Default::default()})).expect("device");
    let mk=|d:&[f32]|device.create_buffer_init(&wgpu::util::BufferInitDescriptor{label:None,contents:bytemuck::cast_slice(d),usage:wgpu::BufferUsages::STORAGE});
    let qb=mk(&q); let kb=mk(&k); let vb=mk(&v);
    let sn=H*Q*S;
    let scores=device.create_buffer(&wgpu::BufferDescriptor{label:None,size:(sn*4) as u64,usage:wgpu::BufferUsages::STORAGE,mapped_at_creation:false});
    let probs=device.create_buffer(&wgpu::BufferDescriptor{label:None,size:(sn*4) as u64,usage:wgpu::BufferUsages::STORAGE,mapped_at_creation:false});
    let ob=device.create_buffer(&wgpu::BufferDescriptor{label:None,size:(Q*H*D*4) as u64,usage:wgpu::BufferUsages::STORAGE|wgpu::BufferUsages::COPY_SRC,mapped_at_creation:false});
    let rb=device.create_buffer(&wgpu::BufferDescriptor{label:None,size:(Q*H*D*4) as u64,usage:wgpu::BufferUsages::COPY_DST|wgpu::BufferUsages::MAP_READ,mapped_at_creation:false});
    let smod=device.create_shader_module(wgpu::ShaderModuleDescriptor{label:None,source:wgpu::ShaderSource::Wgsl(Cow::Borrowed(SCORE_WGSL))});
    let mmod=device.create_shader_module(wgpu::ShaderModuleDescriptor{label:None,source:wgpu::ShaderSource::Wgsl(Cow::Borrowed(SOFT_WGSL))});
    let pmod=device.create_shader_module(wgpu::ShaderModuleDescriptor{label:None,source:wgpu::ShaderSource::Wgsl(Cow::Borrowed(PV_WGSL))});
    let sp=device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor{label:None,layout:None,module:&smod,entry_point:Some("main"),compilation_options:Default::default(),cache:None});
    let mp=device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor{label:None,layout:None,module:&mmod,entry_point:Some("main"),compilation_options:Default::default(),cache:None});
    let pp=device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor{label:None,layout:None,module:&pmod,entry_point:Some("main"),compilation_options:Default::default(),cache:None});
    let sbg=device.create_bind_group(&wgpu::BindGroupDescriptor{label:None,layout:&sp.get_bind_group_layout(0),entries:&[
        wgpu::BindGroupEntry{binding:0,resource:qb.as_entire_binding()},wgpu::BindGroupEntry{binding:1,resource:kb.as_entire_binding()},wgpu::BindGroupEntry{binding:2,resource:scores.as_entire_binding()}]});
    let mbg=device.create_bind_group(&wgpu::BindGroupDescriptor{label:None,layout:&mp.get_bind_group_layout(0),entries:&[
        wgpu::BindGroupEntry{binding:0,resource:scores.as_entire_binding()},wgpu::BindGroupEntry{binding:1,resource:probs.as_entire_binding()}]});
    let pbg=device.create_bind_group(&wgpu::BindGroupDescriptor{label:None,layout:&pp.get_bind_group_layout(0),entries:&[
        wgpu::BindGroupEntry{binding:0,resource:probs.as_entire_binding()},wgpu::BindGroupEntry{binding:1,resource:vb.as_entire_binding()},wgpu::BindGroupEntry{binding:2,resource:ob.as_entire_binding()}]});
    let mut enc=device.create_command_encoder(&wgpu::CommandEncoderDescriptor{label:None});
    {let mut p=enc.begin_compute_pass(&wgpu::ComputePassDescriptor{label:None,timestamp_writes:None});p.set_pipeline(&sp);p.set_bind_group(0,&sbg,&[]);p.dispatch_workgroups(((sn+255)/256) as u32,1,1);}
    {let mut p=enc.begin_compute_pass(&wgpu::ComputePassDescriptor{label:None,timestamp_writes:None});p.set_pipeline(&mp);p.set_bind_group(0,&mbg,&[]);p.dispatch_workgroups((H*Q) as u32,1,1);}
    {let mut p=enc.begin_compute_pass(&wgpu::ComputePassDescriptor{label:None,timestamp_writes:None});p.set_pipeline(&pp);p.set_bind_group(0,&pbg,&[]);p.dispatch_workgroups(((H*Q*D+255)/256) as u32,1,1);}
    enc.copy_buffer_to_buffer(&ob,0,&rb,0,(Q*H*D*4) as u64);
    queue.submit(Some(enc.finish()));
    let slice=rb.slice(..);slice.map_async(wgpu::MapMode::Read,|r|r.expect("map"));device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
    let data=slice.get_mapped_range();let got:&[f32]=bytemuck::cast_slice(&data);
    let mut max_err=0.0f32;for h in 0..H{for i in 0..2{for d in 0..D{max_err=max_err.max((got[(i*H+h)*D+d]-expect[(i*H+h)*D+d]).abs());}}}
    if let Ok(path)=std::env::var("CROSS_OUT") { std::fs::write(path, bytemuck::cast_slice(got)).expect("write out"); }
    println!("cross26-f32 H={H} Q={Q} S={S} D={D} first2_max_abs_err={max_err:.3e}");
    println!("VULKAN CROSS512 OK");
}






