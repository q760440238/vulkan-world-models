use bytemuck::{Pod, Zeroable};
use half::f16;
use std::borrow::Cow;
use wgpu::util::DeviceExt;

const CIN: usize = 6;
const COUT: usize = 1536;
const H: usize = 60;
const W: usize = 104;
const OH: usize = 30;
const OW: usize = 52;
const TOK: usize = OH * OW;

const WGSL: &str = r#"
enable f16;
const CIN: u32 = 6u; const COUT: u32 = 1536u;
const H: u32 = 60u; const W: u32 = 104u;
const OH: u32 = 30u; const OW: u32 = 52u;
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f16>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= (OH * OW) * COUT) { return; }
    let tok = idx / COUT; let co = idx % COUT;
    let oh = tok / OW; let ow = tok % OW;
    var acc = b[co];
    for (var ci = 0u; ci < CIN; ci = ci + 1u) {
        for (var kh = 0u; kh < 2u; kh = kh + 1u) {
            for (var kw = 0u; kw < 2u; kw = kw + 1u) {
                let ih = oh * 2u + kh; let iw = ow * 2u + kw;
                acc = acc + x[(ci * H + ih) * W + iw] * w[((co * CIN + ci) * 2u + kh) * 2u + kw];
            }
        }
    }
    y[idx] = f16(acc);
}
"#;

fn main() {
    let rawx = std::fs::read("D:/3DWord/itseez3d/ai-lab/lingbot-world/port/wgpu-plucker.bin").expect("plucker");
    let x: Vec<f32> = bytemuck::cast_slice::<u8, f32>(&rawx)[..CIN * H * W].to_vec();
    let mut w = vec![0.0f32; COUT * CIN * 4];
    let mut b = vec![0.0f32; COUT];
    for i in 0..w.len() { w[i] = ((i * 17 % 97) as f32 - 48.0) / 41.0; }
    for i in 0..COUT { b[i] = ((i * 7 % 11) as f32 - 5.0) * 0.01; }
    let mut expect = vec![0.0f32; 2 * COUT];
    for tok in 0..2 { let oh=tok/OW; let ow=tok%OW;
        for co in 0..COUT { let mut acc=b[co];
            for ci in 0..CIN { for kh in 0..2 { for kw in 0..2 {
                acc += x[(ci*H + oh*2+kh)*W + ow*2+kw] * w[((co*CIN+ci)*2+kh)*2+kw];
            }}}
            expect[tok*COUT+co]=f16::from_f32(acc).to_f32();
        }
    }
    let instance=wgpu::Instance::new(&wgpu::InstanceDescriptor{backends:wgpu::Backends::VULKAN,..Default::default()});
    let adapter=pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions{power_preference:wgpu::PowerPreference::HighPerformance,compatible_surface:None,force_fallback_adapter:false})).expect("no adapter");
    let info=adapter.get_info(); println!("adapter: {} | backend: {:?}",info.name,info.backend);
    let (device,queue)=pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor{label:Some("cam-patch"),required_features:wgpu::Features::SHADER_F16,required_limits:wgpu::Limits::downlevel_defaults(),..Default::default()})).expect("device");
    let mk=|d:&[f32]|device.create_buffer_init(&wgpu::util::BufferInitDescriptor{label:None,contents:bytemuck::cast_slice(d),usage:wgpu::BufferUsages::STORAGE});
    let xb=mk(&x); let wb=mk(&w); let bb=mk(&b);
    let yb=device.create_buffer(&wgpu::BufferDescriptor{label:None,size:(TOK*COUT*2) as u64,usage:wgpu::BufferUsages::STORAGE|wgpu::BufferUsages::COPY_SRC,mapped_at_creation:false});
    let rb=device.create_buffer(&wgpu::BufferDescriptor{label:None,size:(TOK*COUT*2) as u64,usage:wgpu::BufferUsages::COPY_DST|wgpu::BufferUsages::MAP_READ,mapped_at_creation:false});
    let module=device.create_shader_module(wgpu::ShaderModuleDescriptor{label:None,source:wgpu::ShaderSource::Wgsl(Cow::Borrowed(WGSL))});
    let pipe=device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor{label:None,layout:None,module:&module,entry_point:Some("main"),compilation_options:Default::default(),cache:None});
    let bg=device.create_bind_group(&wgpu::BindGroupDescriptor{label:None,layout:&pipe.get_bind_group_layout(0),entries:&[
        wgpu::BindGroupEntry{binding:0,resource:xb.as_entire_binding()},
        wgpu::BindGroupEntry{binding:1,resource:wb.as_entire_binding()},
        wgpu::BindGroupEntry{binding:2,resource:bb.as_entire_binding()},
        wgpu::BindGroupEntry{binding:3,resource:yb.as_entire_binding()}]});
    let mut enc=device.create_command_encoder(&wgpu::CommandEncoderDescriptor{label:None});
    {let mut p=enc.begin_compute_pass(&wgpu::ComputePassDescriptor{label:None,timestamp_writes:None});p.set_pipeline(&pipe);p.set_bind_group(0,&bg,&[]);p.dispatch_workgroups(((TOK*COUT+63)/64) as u32,1,1);}
    enc.copy_buffer_to_buffer(&yb,0,&rb,0,(TOK*COUT*2) as u64);
    queue.submit(Some(enc.finish()));
    let slice=rb.slice(..);slice.map_async(wgpu::MapMode::Read,|r|r.expect("map"));device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
    let data=slice.get_mapped_range();let got:&[f16]=bytemuck::cast_slice(&data);
    let mut max_err=0.0f32; for i in 0..2*COUT { max_err=max_err.max((got[i].to_f32()-expect[i]).abs()); }
    std::fs::write("D:/3DWord/itseez3d/ai-lab/lingbot-world/port/wgpu-camfeat.bin",bytemuck::cast_slice(got)).expect("write");
    println!("cam-patch TOK={TOK} COUT={COUT} first2_max_abs_err={max_err:.3e}");
    assert!(max_err<5e-2,"cam patch mismatch");
    println!("VULKAN CAM PATCH OK");
}

