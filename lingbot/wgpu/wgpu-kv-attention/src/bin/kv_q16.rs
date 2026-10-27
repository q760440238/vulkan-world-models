use bytemuck::{Pod, Zeroable};
use half::f16;
use std::borrow::Cow;
use wgpu::util::DeviceExt;

const H: usize = 12;
const Q: usize = 16;
const D: usize = 128;
const S: usize = 24 * 1560;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Params { h: u32, q: u32, s: u32, d: u32 }

const SCORE_WGSL: &str = r#"
enable f16;
struct P { h: u32, q: u32, s: u32, d: u32 };
@group(0) @binding(0) var<storage, read> qv: array<f32>;
@group(0) @binding(1) var<storage, read> kv: array<f16>;
@group(0) @binding(2) var<storage, read_write> scores: array<f32>;
@group(0) @binding(3) var<uniform> p: P;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let g = gid.x;
    if (g >= p.h * p.q * p.s) { return; }
    let row = g / p.s; let j = g % p.s;
    let h = row / p.q; let i = row % p.q;
    var dot = 0.0;
    for (var d = 0u; d < p.d; d = d + 1u) {
        dot = dot + qv[(h * p.q + i) * p.d + d] * f32(kv[(h * p.s + j) * p.d + d]);
    }
    scores[g] = dot / sqrt(f32(p.d));
}
"#;
const SOFT_WGSL: &str = r#"
struct P { h: u32, q: u32, s: u32, d: u32 };
@group(0) @binding(0) var<storage, read> scores: array<f32>;
@group(0) @binding(1) var<storage, read_write> probs: array<f32>;
@group(0) @binding(2) var<uniform> p: P;
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if (row >= p.h * p.q) { return; }
    var maxs = -1e30;
    for (var j = 0u; j < p.s; j = j + 1u) { maxs = max(maxs, scores[row * p.s + j]); }
    var sum = 0.0;
    for (var j = 0u; j < p.s; j = j + 1u) { sum = sum + exp(scores[row * p.s + j] - maxs); }
    for (var j = 0u; j < p.s; j = j + 1u) { probs[row * p.s + j] = exp(scores[row * p.s + j] - maxs) / sum; }
}
"#;
const PV_WGSL: &str = r#"
enable f16;
struct P { h: u32, q: u32, s: u32, d: u32 };
@group(0) @binding(0) var<storage, read> probs: array<f32>;
@group(0) @binding(1) var<storage, read> vv: array<f16>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> p: P;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let g = gid.x;
    if (g >= p.h * p.q * p.d) { return; }
    let row = g / p.d; let d = g % p.d;
    let h = row / p.q;
    var acc = 0.0;
    for (var j = 0u; j < p.s; j = j + 1u) {
        acc = acc + probs[row * p.s + j] * f32(vv[(h * p.s + j) * p.d + d]);
    }
    out[g] = acc;
}
"#;

fn main() {
    println!("allocating Q={Q} S={S}...");
    let mut q = vec![0.0f32; H * Q * D];
    let mut k = vec![f16::from_f32(0.0); H * S * D];
    let mut v = vec![f16::from_f32(0.0); H * S * D];
    for i in 0..q.len() { q[i] = ((i * 13 % 71) as f32 - 35.0) / 19.0; }
    for i in 0..k.len() {
        k[i] = f16::from_f32(((i * 17 % 97) as f32 - 48.0) / 37.0);
        v[i] = f16::from_f32(((i * 23 % 89) as f32 - 44.0) / 41.0);
    }
    let sink=6*1560; let local=18*1560; assert_eq!(sink+local,S);
    let newn=1560*D;
    let mut nk=vec![f16::from_f32(0.0);H*newn]; let mut nv=vec![f16::from_f32(0.0);H*newn];
    for i in 0..H*newn { nk[i]=f16::from_f32(((i*31%73) as f32-36.0)/43.0); nv[i]=f16::from_f32(((i*37%79) as f32-39.0)/47.0); }
    let mut k2=vec![f16::from_f32(0.0);H*S*D]; let mut v2=vec![f16::from_f32(0.0);H*S*D];
    for h in 0..H { let src=h*S*D; let dst=h*S*D;
        k2[dst..dst+sink*D].copy_from_slice(&k[src..src+sink*D]); v2[dst..dst+sink*D].copy_from_slice(&v[src..src+sink*D]);
        let ls=src+(sink+1560)*D; let ln=(local-1560)*D;
        k2[dst+sink*D..dst+sink*D+ln].copy_from_slice(&k[ls..ls+ln]); v2[dst+sink*D..dst+sink*D+ln].copy_from_slice(&v[ls..ls+ln]);
        let tail=dst+(sink+local-1560)*D;
        k2[tail..tail+newn].copy_from_slice(&nk[h*newn..(h+1)*newn]); v2[tail..tail+newn].copy_from_slice(&nv[h*newn..(h+1)*newn]);
    }
    k=k2; v=v2;
    let expect = cpu_q(&q,&k,&v);
    let instance=wgpu::Instance::new(&wgpu::InstanceDescriptor{backends:wgpu::Backends::VULKAN,..Default::default()});
    let adapter=pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions{power_preference:wgpu::PowerPreference::HighPerformance,compatible_surface:None,force_fallback_adapter:false})).expect("no adapter");
    let info=adapter.get_info(); println!("adapter: {} | backend: {:?}",info.name,info.backend);
    let mut limits=wgpu::Limits::downlevel_defaults(); limits.max_storage_buffer_binding_size=128<<20;
    let (device,queue)=pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor{label:Some("kv-q16"),required_features:wgpu::Features::SHADER_F16,required_limits:limits,..Default::default()})).expect("device");
    let mk=|d:&[f32]|device.create_buffer_init(&wgpu::util::BufferInitDescriptor{label:None,contents:bytemuck::cast_slice(d),usage:wgpu::BufferUsages::STORAGE});
    let mk16=|d:&[f16]|device.create_buffer_init(&wgpu::util::BufferInitDescriptor{label:None,contents:bytemuck::cast_slice(d),usage:wgpu::BufferUsages::STORAGE});
    let qb=mk(&q); let kb=mk16(&k); let vb=mk16(&v);
    let sn=H*Q*S;
    let scores=device.create_buffer(&wgpu::BufferDescriptor{label:None,size:(sn*4) as u64,usage:wgpu::BufferUsages::STORAGE,mapped_at_creation:false});
    let probs=device.create_buffer(&wgpu::BufferDescriptor{label:None,size:(sn*4) as u64,usage:wgpu::BufferUsages::STORAGE,mapped_at_creation:false});
    let ob=device.create_buffer(&wgpu::BufferDescriptor{label:None,size:(H*Q*D*4) as u64,usage:wgpu::BufferUsages::STORAGE|wgpu::BufferUsages::COPY_SRC,mapped_at_creation:false});
    let rb=device.create_buffer(&wgpu::BufferDescriptor{label:None,size:(H*Q*D*4) as u64,usage:wgpu::BufferUsages::COPY_DST|wgpu::BufferUsages::MAP_READ,mapped_at_creation:false});
    let params=Params{h:H as u32,q:Q as u32,s:S as u32,d:D as u32};
    let pb=device.create_buffer_init(&wgpu::util::BufferInitDescriptor{label:None,contents:bytemuck::bytes_of(&params),usage:wgpu::BufferUsages::UNIFORM});
    let smod=device.create_shader_module(wgpu::ShaderModuleDescriptor{label:None,source:wgpu::ShaderSource::Wgsl(Cow::Borrowed(SCORE_WGSL))});
    let mmod=device.create_shader_module(wgpu::ShaderModuleDescriptor{label:None,source:wgpu::ShaderSource::Wgsl(Cow::Borrowed(SOFT_WGSL))});
    let pmod=device.create_shader_module(wgpu::ShaderModuleDescriptor{label:None,source:wgpu::ShaderSource::Wgsl(Cow::Borrowed(PV_WGSL))});
    let sp=device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor{label:None,layout:None,module:&smod,entry_point:Some("main"),compilation_options:Default::default(),cache:None});
    let mp=device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor{label:None,layout:None,module:&mmod,entry_point:Some("main"),compilation_options:Default::default(),cache:None});
    let pp=device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor{label:None,layout:None,module:&pmod,entry_point:Some("main"),compilation_options:Default::default(),cache:None});
    let sbg=device.create_bind_group(&wgpu::BindGroupDescriptor{label:None,layout:&sp.get_bind_group_layout(0),entries:&[
        wgpu::BindGroupEntry{binding:0,resource:qb.as_entire_binding()},wgpu::BindGroupEntry{binding:1,resource:kb.as_entire_binding()},wgpu::BindGroupEntry{binding:2,resource:scores.as_entire_binding()},wgpu::BindGroupEntry{binding:3,resource:pb.as_entire_binding()}]});
    let mbg=device.create_bind_group(&wgpu::BindGroupDescriptor{label:None,layout:&mp.get_bind_group_layout(0),entries:&[
        wgpu::BindGroupEntry{binding:0,resource:scores.as_entire_binding()},wgpu::BindGroupEntry{binding:1,resource:probs.as_entire_binding()},wgpu::BindGroupEntry{binding:2,resource:pb.as_entire_binding()}]});
    let pbg=device.create_bind_group(&wgpu::BindGroupDescriptor{label:None,layout:&pp.get_bind_group_layout(0),entries:&[
        wgpu::BindGroupEntry{binding:0,resource:probs.as_entire_binding()},wgpu::BindGroupEntry{binding:1,resource:vb.as_entire_binding()},wgpu::BindGroupEntry{binding:2,resource:ob.as_entire_binding()},wgpu::BindGroupEntry{binding:3,resource:pb.as_entire_binding()}]});
    let mut enc=device.create_command_encoder(&wgpu::CommandEncoderDescriptor{label:None});
    {let mut p=enc.begin_compute_pass(&wgpu::ComputePassDescriptor{label:None,timestamp_writes:None});p.set_pipeline(&sp);p.set_bind_group(0,&sbg,&[]);p.dispatch_workgroups(((sn+255)/256) as u32,1,1);}
    {let mut p=enc.begin_compute_pass(&wgpu::ComputePassDescriptor{label:None,timestamp_writes:None});p.set_pipeline(&mp);p.set_bind_group(0,&mbg,&[]);p.dispatch_workgroups((H*Q) as u32,1,1);}
    {let mut p=enc.begin_compute_pass(&wgpu::ComputePassDescriptor{label:None,timestamp_writes:None});p.set_pipeline(&pp);p.set_bind_group(0,&pbg,&[]);p.dispatch_workgroups(((H*Q*D+255)/256) as u32,1,1);}
    enc.copy_buffer_to_buffer(&ob,0,&rb,0,(H*Q*D*4) as u64);
    let t0=std::time::Instant::now(); queue.submit(Some(enc.finish()));
    let slice=rb.slice(..);slice.map_async(wgpu::MapMode::Read,|r|r.expect("map"));device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
    let data=slice.get_mapped_range(); let got:&[f32]=bytemuck::cast_slice(&data);
    let mut max_err=0.0f32; for h in 0..H { for i in 0..2 { for d in 0..D { max_err=max_err.max((got[(h*Q+i)*D+d]-expect[(h*Q+i)*D+d]).abs()); } } }
    println!("kv-q16 H={H} Q={Q} S={S} D={D} gpu_ms={:.3} first2_max_abs_err={max_err:.3e}",t0.elapsed().as_secs_f64()*1000.0);
    assert!(max_err<1e-3,"kv q16 mismatch");
    println!("VULKAN KV Q16 OK");
}

fn cpu_q(q:&[f32],k:&[f16],v:&[f16])->Vec<f32>{
    let mut out=vec![0.0f32;H*Q*D];
    for h in 0..H { for i in 0..Q {
        let mut scores=vec![0.0f32;S]; let mut maxs=f32::NEG_INFINITY;
        for j in 0..S { let mut dot=0.0f32; for d in 0..D { dot+=q[(h*Q+i)*D+d]*k[(h*S+j)*D+d].to_f32(); } scores[j]=dot/(D as f32).sqrt(); maxs=maxs.max(scores[j]); }
        let sum:f32=scores.iter().map(|s|(s-maxs).exp()).sum();
        for d in 0..D { let mut acc=0.0f32; for j in 0..S { acc+=(scores[j]-maxs).exp()/sum*v[(h*S+j)*D+d].to_f32(); } out[(h*Q+i)*D+d]=acc; }
    }}
    out
}
