use bytemuck::Pod; use bytemuck::Zeroable; use std::borrow::Cow; use wgpu::util::DeviceExt;
const M:usize=1560; const C:usize=1536;
const WGSL:&str=r#"
const C:u32=1536u;
@group(0) @binding(0) var<storage,read> x:array<f32>;
@group(0) @binding(1) var<storage,read> g:array<f32>;
@group(0) @binding(2) var<storage,read> b:array<f32>;
@group(0) @binding(3) var<storage,read_write> y:array<f32>;
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) gid:vec3<u32>){
 let t=gid.x; if(t>=1560u){return;}
 var m=0.0; for(var c=0u;c<C;c=c+1u){m=m+x[t*C+c];} m=m/f32(C);
 var v=0.0; for(var c=0u;c<C;c=c+1u){let d=x[t*C+c]-m; v=v+d*d;} v=v/f32(C);
 let inv=inverseSqrt(v+1e-6);
 for(var c=0u;c<C;c=c+1u){y[t*C+c]=(x[t*C+c]-m)*inv*g[c]+b[c];}
}
"#;
fn main(){
 let ip=std::env::var("LN_INPUT").unwrap(); let gp=std::env::var("LN_GAMMA").unwrap(); let bp=std::env::var("LN_BETA").unwrap(); let op=std::env::var("LN_OUT").unwrap();
 let x:Vec<f32>=bytemuck::cast_slice::<u8,f32>(&std::fs::read(ip).unwrap())[..M*C].to_vec();
 let g:Vec<f32>=bytemuck::cast_slice::<u8,f32>(&std::fs::read(gp).unwrap())[..C].to_vec();
 let b:Vec<f32>=bytemuck::cast_slice::<u8,f32>(&std::fs::read(bp).unwrap())[..C].to_vec();
 let mut ex=vec![0.0f32;2*C];
 for t in 0..2{let m:f32=(0..C).map(|c|x[t*C+c]).sum::<f32>()/C as f32;let v:f32=(0..C).map(|c|(x[t*C+c]-m)*(x[t*C+c]-m)).sum::<f32>()/C as f32;let inv=1.0/(v+1e-6).sqrt();for c in 0..C{ex[t*C+c]=(x[t*C+c]-m)*inv*g[c]+b[c];}}
 let inst=wgpu::Instance::new(&wgpu::InstanceDescriptor{backends:wgpu::Backends::VULKAN,..Default::default()});
 let ad=pollster::block_on(inst.request_adapter(&wgpu::RequestAdapterOptions{power_preference:wgpu::PowerPreference::HighPerformance,compatible_surface:None,force_fallback_adapter:false})).unwrap();
 let (dev,q)=pollster::block_on(ad.request_device(&wgpu::DeviceDescriptor{label:None,required_features:wgpu::Features::empty(),required_limits:wgpu::Limits::downlevel_defaults(),..Default::default()})).unwrap();
 let mk=|d:&[f32]|dev.create_buffer_init(&wgpu::util::BufferInitDescriptor{label:None,contents:bytemuck::cast_slice(d),usage:wgpu::BufferUsages::STORAGE});
 let xb=mk(&x);let gb=mk(&g);let bb=mk(&b);
 let yb=dev.create_buffer(&wgpu::BufferDescriptor{label:None,size:(M*C*4) as u64,usage:wgpu::BufferUsages::STORAGE|wgpu::BufferUsages::COPY_SRC,mapped_at_creation:false});
 let rb=dev.create_buffer(&wgpu::BufferDescriptor{label:None,size:(M*C*4) as u64,usage:wgpu::BufferUsages::COPY_DST|wgpu::BufferUsages::MAP_READ,mapped_at_creation:false});
 let md=dev.create_shader_module(wgpu::ShaderModuleDescriptor{label:None,source:wgpu::ShaderSource::Wgsl(Cow::Borrowed(WGSL))});
 let pipe=dev.create_compute_pipeline(&wgpu::ComputePipelineDescriptor{label:None,layout:None,module:&md,entry_point:Some("main"),compilation_options:Default::default(),cache:None});
 let bg=dev.create_bind_group(&wgpu::BindGroupDescriptor{label:None,layout:&pipe.get_bind_group_layout(0),entries:&[wgpu::BindGroupEntry{binding:0,resource:xb.as_entire_binding()},wgpu::BindGroupEntry{binding:1,resource:gb.as_entire_binding()},wgpu::BindGroupEntry{binding:2,resource:bb.as_entire_binding()},wgpu::BindGroupEntry{binding:3,resource:yb.as_entire_binding()}]});
 let mut en=dev.create_command_encoder(&wgpu::CommandEncoderDescriptor{label:None});{let mut p=en.begin_compute_pass(&wgpu::ComputePassDescriptor{label:None,timestamp_writes:None});p.set_pipeline(&pipe);p.set_bind_group(0,&bg,&[]);p.dispatch_workgroups(M as u32,1,1);}en.copy_buffer_to_buffer(&yb,0,&rb,0,(M*C*4) as u64);q.submit(Some(en.finish()));
 let sl=rb.slice(..);sl.map_async(wgpu::MapMode::Read,|r|r.unwrap());dev.poll(wgpu::PollType::wait_indefinitely()).unwrap();let d=sl.get_mapped_range();let y:&[f32]=bytemuck::cast_slice(&d);
 let mut e=0.0f32;for i in 0..2*C{e=e.max((y[i]-ex[i]).abs());}std::fs::write(op,bytemuck::cast_slice(&y[..M*C])).unwrap();println!("ln-f32 maxerr {e:.3e}");
}
