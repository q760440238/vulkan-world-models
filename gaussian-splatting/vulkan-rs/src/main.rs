use bytemuck::{Pod,Zeroable}; use std::borrow::Cow; use wgpu::util::DeviceExt;
const W:usize=832; const H:usize=480;
#[repr(C)] #[derive(Copy,Clone,Pod,Zeroable)]
struct Inst{c:[f32;2],a1:[f32;2],a2:[f32;2],pad:[f32;2],col:[f32;4]}
const WGSL:&str=r#"
struct Inst{c:vec2f,a1:vec2f,a2:vec2f,pad:vec2f,col:vec4f};
@group(0)@binding(0) var<storage,read> insts:array<Inst>;
struct VOut{@builtin(position) p:vec4f,@location(0) l:vec2f,@location(1) c:vec4f};
@vertex fn vs(@builtin(vertex_index) vi:u32,@builtin(instance_index) ii:u32)->VOut{
 var cs=array<vec2f,6>(vec2f(-1.,-1.),vec2f(1.,-1.),vec2f(-1.,1.),vec2f(-1.,1.),vec2f(1.,-1.),vec2f(1.,1.));
 let q=cs[vi]; let ins=insts[ii]; let p=ins.c+q.x*ins.a1+q.y*ins.a2;
 var o:VOut; o.p=vec4f(p.x/416.-1.,1.-p.y/240.,0.5,1.); o.l=q; o.c=ins.col; return o;
}
@fragment fn fs(i:VOut)->@location(0) vec4f{
 let a=i.c.a*exp(-4.5*dot(i.l,i.l)); if(a<0.003){discard;} return vec4f(i.c.rgb,a);
}
"#;
fn main(){
 let args:Vec<String>=std::env::args().collect(); let path=args.get(1).map(|s|s.as_str()).unwrap_or(r"D:\3DWord\itseez3d\ai-lab\ml-sharp\output\photo.ply");
 let raw=std::fs::read(path).unwrap(); let hs=find(&raw,b"end_header
").unwrap()+11; let head=String::from_utf8_lossy(&raw[..hs]);
 let n:usize=head.lines().find(|l|l.starts_with("element vertex")).unwrap().split_whitespace().nth(2).unwrap().parse().unwrap();
 let floats:Vec<f32>=raw[hs..hs+n*14*4].chunks_exact(4).map(|b| f32::from_le_bytes([b[0],b[1],b[2],b[3]])).collect();
 let mut ps=Vec::with_capacity(n); for i in 0..n{let o=i*14; ps.push([floats[o],floats[o+1],floats[o+2]]);} 
 let (mut mn,mut mx)=([f32::MAX;3],[f32::MIN;3]); for p in &ps{for k in 0..3{mn[k]=mn[k].min(p[k]);mx[k]=mx[k].max(p[k]);}}
 let ctr=[(mn[0]+mx[0])/2.,(mn[1]+mx[1])/2.,(mn[2]+mx[2])/2.]; let mut rad=0.0f32; for p in &ps{let d=(p[0]-ctr[0]).powi(2)+(p[1]-ctr[1]).powi(2)+(p[2]-ctr[2]).powi(2);rad=rad.max(d);} let rad=rad.sqrt();
 let yaw=args.get(2).and_then(|s|s.parse().ok()).unwrap_or(0.0f32); let (cy,sy)=(yaw.cos(),yaw.sin()); let orbit=args.get(5).map(|s|s=="orbit").unwrap_or(false); let dist=1.8*rad; let eye=if orbit{[ctr[0]+dist*sy,ctr[1]+0.15*rad,ctr[2]+dist*cy]}else{[0.,0.,0.]}; let mut rv=[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]; if orbit{let ex=ctr[0]-eye[0];let ey=ctr[1]-eye[1];let ez=ctr[2]-eye[2];let el=(ex*ex+ey*ey+ez*ez).sqrt();let f=[ex/el,ey/el,ez/el];let rx=-f[2];let rz=f[0];let rl=(rx*rx+rz*rz).sqrt();let r=[rx/rl,0.,rz/rl];let u=[f[1]*r[2]-f[2]*r[1],f[2]*r[0]-f[0]*r[2],f[0]*r[1]-f[1]*r[0]];rv=[r,u,f];}
 let covmin=args.get(4).and_then(|s|s.parse().ok()).unwrap_or(0.10f32);
 let fx=416.0f32; let fy=416.0f32; let cx0=416.0f32; let cy0=240.0f32;
 let mut tmp:Vec<(f32,Inst)>=Vec::with_capacity(n);
 for i in 0..n{
  let o=i*14; let dc=[floats[o+3],floats[o+4],floats[o+5]]; let alpha=(1.0/(1.0+(-floats[o+6]).exp())*2.0).min(1.0);
  let sc=[floats[o+7].exp(),floats[o+8].exp(),floats[o+9].exp()];
  let (mut w,mut x,mut y,mut z)=(floats[o+10],floats[o+11],floats[o+12],floats[o+13]); let qn=(w*w+x*x+y*y+z*z).sqrt(); if qn<1e-8{w=1.;x=0.;y=0.;z=0.;}else{w/=qn;x/=qn;y/=qn;z/=qn;}
  let r=[[1.-2.*(y*y+z*z),2.*(x*y-w*z),2.*(x*z+w*y)],[2.*(x*y+w*z),1.-2.*(x*x+z*z),2.*(y*z-w*x)],[2.*(x*z-w*y),2.*(y*z+w*x),1.-2.*(x*x+y*y)]];
  let mut cov=[[0f32;3];3]; for a in 0..3{for b in 0..3{let mut t=0.;for k in 0..3{t+=r[a][k]*sc[k]*sc[k]*r[b][k];}cov[a][b]=t;}}
  let xc=cy*ps[i][0]-sy*ps[i][2]; let yc=ps[i][1]; let zc=sy*ps[i][0]+cy*ps[i][2];
  if zc<0.05{continue;}
  let u=fx*xc/zc+cx0; let v=fy*yc/zc+cy0; if u < -50.||u>W as f32+50.||v < -50.||v>H as f32+50.{continue;}
  let cxx=cov[0][0]; let cxz=cov[0][2]; let czz=cov[2][2];
  let r00=cy*cy*cxx-2.*cy*sy*cxz+sy*sy*czz;
  let r02=cy*sy*(cxx-czz)+(cy*cy-sy*sy)*cxz;
  let r22=sy*sy*cxx+2.*sy*cy*cxz+cy*cy*czz;
  let r01=cy*cov[0][1]-sy*cov[2][1];
  let r12=sy*cov[0][1]+cy*cov[2][1];
  let iz=1./zc; let j00=fx*iz; let j02=-fx*xc*iz*iz; let j11=fy*iz; let j12=-fy*yc*iz*iz;
  let a2=j00*j00*r00+2.*j00*j02*r02+j02*j02*r22;
  let b2=j00*j11*r01+j00*j12*r02+j02*j11*r12+j02*j12*r22;
  let c2=j11*j11*cov[1][1]+2.*j11*j12*r12+j12*j12*r22;
  let mid=(a2+c2)/2.; let d=(((a2-c2)/2.).powi(2)+b2*b2).sqrt(); let l1=(mid+d).max(0.01);let l2=(mid-d).max(0.01);
  let (mut v1x,mut v1y)=(b2,l1-a2); if v1x.abs()+v1y.abs()<1e-6{v1x=1.;v1y=0.;} let nn=(v1x*v1x+v1y*v1y).sqrt();v1x/=nn;v1y/=nn; let v2x=-v1y;let v2y=v1x;
  let s1=((3.*l1).sqrt()*2.0).clamp(covmin,12.); let s2=((3.*l2).sqrt()*2.0).clamp(covmin,12.); if !s1.is_finite()||!s2.is_finite()||!u.is_finite()||!v.is_finite()||alpha<0.003{continue;}
  let col=[dc[0].clamp(0.,1.),dc[1].clamp(0.,1.),dc[2].clamp(0.,1.)];
  tmp.push((zc,Inst{c:[u,v],a1:[v1x*s1,v1y*s1],a2:[v2x*s2,v2y*s2],pad:[0.,0.],col:[col[0],col[1],col[2],alpha]}));
 }
 tmp.sort_by(|a,b|b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)); let inst:Vec<Inst>=tmp.into_iter().map(|x|x.1).collect(); println!("gaussians={} visible={}",n,inst.len());

 let instance=wgpu::Instance::new(&wgpu::InstanceDescriptor{backends:wgpu::Backends::VULKAN,..Default::default()}); let ad=pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions{power_preference:wgpu::PowerPreference::HighPerformance,compatible_surface:None,force_fallback_adapter:false})).unwrap(); let info=ad.get_info();println!("{} {:?}",info.name,info.backend);
 let mut lim=wgpu::Limits::downlevel_defaults();lim.max_storage_buffer_binding_size=512<<20;
 let (dev,q)=pollster::block_on(ad.request_device(&wgpu::DeviceDescriptor{label:None,required_features:wgpu::Features::empty(),required_limits:lim,..Default::default()})).unwrap();
 let ib=dev.create_buffer_init(&wgpu::util::BufferInitDescriptor{label:None,contents:bytemuck::cast_slice(&inst),usage:wgpu::BufferUsages::STORAGE});
 let tex=dev.create_texture(&wgpu::TextureDescriptor{label:None,size:wgpu::Extent3d{width:W as u32,height:H as u32,depth_or_array_layers:1},mip_level_count:1,sample_count:1,dimension:wgpu::TextureDimension::D2,format:wgpu::TextureFormat::Rgba8UnormSrgb,usage:wgpu::TextureUsages::RENDER_ATTACHMENT|wgpu::TextureUsages::COPY_SRC,view_formats:&[]});
 let view=tex.create_view(&wgpu::TextureViewDescriptor::default());
 let md=dev.create_shader_module(wgpu::ShaderModuleDescriptor{label:None,source:wgpu::ShaderSource::Wgsl(Cow::Borrowed(WGSL))});
 let pl=dev.create_render_pipeline(&wgpu::RenderPipelineDescriptor{label:None,layout:None,vertex:wgpu::VertexState{module:&md,entry_point:Some("vs"),compilation_options:Default::default(),buffers:&[]},fragment:Some(wgpu::FragmentState{module:&md,entry_point:Some("fs"),compilation_options:Default::default(),targets:&[Some(wgpu::ColorTargetState{format:wgpu::TextureFormat::Rgba8UnormSrgb,blend:Some(wgpu::BlendState::ALPHA_BLENDING),write_mask:wgpu::ColorWrites::ALL})]}),primitive:wgpu::PrimitiveState::default(),depth_stencil:None,multisample:wgpu::MultisampleState::default(),multiview:None,cache:None});
 let bgl=pl.get_bind_group_layout(0); let bg=dev.create_bind_group(&wgpu::BindGroupDescriptor{label:None,layout:&bgl,entries:&[wgpu::BindGroupEntry{binding:0,resource:ib.as_entire_binding()}]});
 let mut en=dev.create_command_encoder(&wgpu::CommandEncoderDescriptor{label:None});
 {let mut rp=en.begin_render_pass(&wgpu::RenderPassDescriptor{label:None,color_attachments:&[Some(wgpu::RenderPassColorAttachment{view:&view,resolve_target:None,depth_slice:None,ops:wgpu::Operations{load:wgpu::LoadOp::Clear(wgpu::Color{r:0.55,g:0.72,b:0.9,a:1.}),store:wgpu::StoreOp::Store}})],depth_stencil_attachment:None,timestamp_writes:None,occlusion_query_set:None});rp.set_pipeline(&pl);rp.set_bind_group(0,&bg,&[]);rp.draw(0..6,0..inst.len() as u32);}
 let bpr=(W*4) as u32; let rb=dev.create_buffer(&wgpu::BufferDescriptor{label:None,size:(bpr*H as u32) as u64,usage:wgpu::BufferUsages::COPY_DST|wgpu::BufferUsages::MAP_READ,mapped_at_creation:false});
 en.copy_texture_to_buffer(wgpu::TexelCopyTextureInfo{texture:&tex,mip_level:0,origin:wgpu::Origin3d::ZERO,aspect:wgpu::TextureAspect::All},wgpu::TexelCopyBufferInfo{buffer:&rb,layout:wgpu::TexelCopyBufferLayout{offset:0,bytes_per_row:Some(bpr),rows_per_image:Some(H as u32)}},wgpu::Extent3d{width:W as u32,height:H as u32,depth_or_array_layers:1});
 q.submit(Some(en.finish()));let sl=rb.slice(..);sl.map_async(wgpu::MapMode::Read,|r|r.unwrap());dev.poll(wgpu::PollType::wait_indefinitely()).unwrap();let d=sl.get_mapped_range();let mut img=Vec::with_capacity(W*H*3);for y in 0..H{for x in 0..W{let o=y*bpr as usize+x*4;img.push(d[o]);img.push(d[o+1]);img.push(d[o+2]);}}
 let outp=args.get(3).map(|s|s.to_string()).unwrap_or(r"D:\3DWord\itseez3d\ai-lab\gaussian-splatting\vulkan-rs\splat.png".to_string()); png_rgb(&outp,W as u32,H as u32,&img); println!("saved {}",outp);
}
fn find(h:&[u8],n:&[u8])->Option<usize>{(0..h.len()).find(|i|h.get(*i..*i+n.len())==Some(n))}












fn crc32(b:&[u8])->u32{let mut c=0xffffffffu32;for &x in b{c^=x as u32;for _ in 0..8{c=if c&1!=0{0xedb88320^(c>>1)}else{c>>1};}}c^0xffffffff}
fn adler32(b:&[u8])->u32{let (mut a,mut d)=(1u32,0u32);for &x in b{a=(a+x as u32)%65521;d=(d+a)%65521;}(d<<16)|a}
fn chunk(out:&mut Vec<u8>,typ:&[u8;4],data:&[u8]){out.extend_from_slice(&(data.len() as u32).to_be_bytes());let mut t=Vec::with_capacity(4+data.len());t.extend_from_slice(typ);t.extend_from_slice(data);out.extend_from_slice(&t);out.extend_from_slice(&crc32(&t).to_be_bytes());}
fn png_rgb(path:&str,w:u32,h:u32,rgb:&[u8]){
 let mut raw=Vec::with_capacity((w as usize*3+1)*h as usize);
 for y in 0..h as usize{raw.push(0);raw.extend_from_slice(&rgb[y*w as usize*3..(y+1)*w as usize*3]);}
 let mut z=vec![0x78,0x01]; let mut off=0;
 while off<raw.len(){let n=(raw.len()-off).min(65535);let last=off+n>=raw.len();z.push(if last{1}else{0});z.extend_from_slice(&(n as u16).to_le_bytes());z.extend_from_slice(&(!(n as u16)).to_le_bytes());z.extend_from_slice(&raw[off..off+n]);off+=n;}
 z.extend_from_slice(&adler32(&raw).to_be_bytes());
 let mut out=vec![137,80,78,71,13,10,26,10]; let mut ihdr=Vec::new();ihdr.extend_from_slice(&w.to_be_bytes());ihdr.extend_from_slice(&h.to_be_bytes());ihdr.extend_from_slice(&[8,2,0,0,0]);chunk(&mut out,b"IHDR",&ihdr);chunk(&mut out,b"IDAT",&z);chunk(&mut out,b"IEND",&[]);std::fs::write(path,out).unwrap();
}






