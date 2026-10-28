use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use wgpu::util::DeviceExt;

pub const F32: u32 = 4;

#[derive(Clone)]
pub struct Buf {
    pub buffer: wgpu::Buffer,
    pub offset: u64,
    pub size: u64,
}

impl Buf {
    pub fn binding(&self) -> wgpu::BufferBinding<'_> {
        wgpu::BufferBinding {
            buffer: &self.buffer,
            offset: self.offset,
            size: std::num::NonZeroU64::new(self.size),
        }
    }
    pub fn slice(&self, offset: u64, size: u64) -> Buf {
        assert!(offset + size <= self.size, "slice out of bounds: {}+{} > {}", offset, size, self.size);
        Buf { buffer: self.buffer.clone(), offset: self.offset + offset, size }
    }
}

pub struct Gpu {
    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub pipelines: HashMap<String, wgpu::ComputePipeline>,
    pub dummy: Buf,
}

impl Gpu {
    pub fn new() -> Self {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..Default::default()
        });
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .expect("no Vulkan adapter");
        let info = adapter.get_info();
        println!("adapter: {} | backend: {:?} | type: {:?}", info.name, info.backend, info.device_type);
        assert_eq!(info.backend, wgpu::Backend::Vulkan, "expected Vulkan backend");
        let limits = adapter.limits();
        println!(
            "limits: max_buffer={} max_storage_binding={} max_wg_storage={} max_wg={} max_invocations={}",
            limits.max_buffer_size,
            limits.max_storage_buffer_binding_size,
            limits.max_compute_workgroup_storage_size,
            limits.max_compute_workgroup_size_x,
            limits.max_compute_invocations_per_workgroup
        );
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("hwm-vk"),
            required_features: wgpu::Features::empty(),
            required_limits: limits.clone(),
            ..Default::default()
        }))
        .expect("device");
        device.on_uncaptured_error(std::sync::Arc::new(|e| panic!("wgpu error: {e:?}")));
        let dummy = {
            let b = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("dummy"),
                contents: &[0u8; 1024],
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });
            Buf { buffer: b, offset: 0, size: 1024 }
        };
        Self { instance, adapter, device, queue, pipelines: HashMap::new(), dummy }
    }

    pub fn add_pipeline(&mut self, name: &str, code: &str) {
        let module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(name),
            source: wgpu::ShaderSource::Wgsl(code.into()),
        });
        let pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(name),
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        self.pipelines.insert(name.to_string(), pipeline);
    }

    pub fn buffer(&self, size: u64, usage: wgpu::BufferUsages) -> Buf {
        let size = ((size + 3) & !3).max(4);
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("buf"),
            size,
            usage,
            mapped_at_creation: false,
        });
        Buf { buffer, offset: 0, size }
    }

    pub fn upload(&self, data: &[u8]) -> Buf {
        let contents = if data.is_empty() { &[0u8, 0, 0, 0][..] } else { data };
        let size = contents.len() as u64;
        let buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("upload"),
            contents,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        });
        Buf { buffer, offset: 0, size }
    }

    /// Dispatch one compute pipeline. `bindings` are the data bindings in order;
    /// the uniform params buffer (if non-empty) is appended as the last binding.
    pub fn dispatch(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        name: &str,
        bindings: &[&Buf],
        params: &[u8],
        workgroups: (u32, u32, u32),
    ) {
        let pipeline = self.pipelines.get(name).unwrap_or_else(|| panic!("unknown pipeline {name}"));
        let pbuf = if params.is_empty() {
            None
        } else {
            let b = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("params"),
                size: 256,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.queue.write_buffer(&b, 0, params);
            Some(Buf { buffer: b, offset: 0, size: 256 })
        };
        let mut entries: Vec<wgpu::BindGroupEntry> = Vec::new();
        for (i, b) in bindings.iter().enumerate() {
            entries.push(wgpu::BindGroupEntry {
                binding: i as u32,
                resource: wgpu::BindingResource::Buffer(b.binding()),
            });
        }
        if let Some(p) = &pbuf {
            entries.push(wgpu::BindGroupEntry {
                binding: bindings.len() as u32,
                resource: wgpu::BindingResource::Buffer(p.binding()),
            });
        }
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &entries,
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(name),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(workgroups.0, workgroups.1, workgroups.2);
    }

    pub fn finish(&self, encoder: wgpu::CommandEncoder) {
        self.queue.submit(Some(encoder.finish()));
    }

    pub fn read(&self, buf: &Buf) -> Vec<u8> {
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: buf.size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("read") });
        encoder.copy_buffer_to_buffer(&buf.buffer, buf.offset, &staging, 0, buf.size);
        self.queue.submit(Some(encoder.finish()));
        let slice = staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, |r| r.expect("map"));
        self.device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
        let data = slice.get_mapped_range().to_vec();
        self.device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
        data
    }

    pub fn read_f32(&self, buf: &Buf) -> Vec<f32> {
        let bytes = self.read(buf);
        bytemuck::cast_slice(&bytes).to_vec()
    }

    pub fn write_f32(&self, buf: &Buf, data: &[f32]) {
        assert!(data.len() * 4 <= buf.size as usize);
        self.queue.write_buffer(&buf.buffer, buf.offset, bytemuck::cast_slice(data));
    }

    pub fn wait(&self) {
        self.device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
    }
}

pub struct WeightStore {
    entries: HashMap<String, (usize, u64, u64, Vec<u32>)>,
    pub chunks: Vec<Buf>,
}

impl WeightStore {
    pub fn open(idx_path: &Path, data_path: &Path, gpu: &Gpu, chunk_size: u64) -> Self {
        Self::open_filtered(idx_path, data_path, gpu, chunk_size, None)
    }

    pub fn open_filtered(
        idx_path: &Path,
        data_path: &Path,
        gpu: &Gpu,
        chunk_size: u64,
        prefixes: Option<&[&str]>,
    ) -> Self {
        let text = std::fs::read_to_string(idx_path).expect("weights.idx");
        let mut file = File::open(data_path).expect("model.safetensors");
        let mut entries = HashMap::new();
        let mut chunks: Vec<Buf> = Vec::new();
        let mut cur: Option<Buf> = None;
        let mut used = 0u64;
        let mut total = 0u64;
        let t0 = std::time::Instant::now();
        let mut count = 0usize;
        for line in text.lines() {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.is_empty() {
                continue;
            }
            let name = parts[0].to_string();
            if let Some(prefixes) = prefixes {
                if !prefixes.iter().any(|p| name.starts_with(p)) {
                    continue;
                }
            }
            let offset: u64 = parts[1].parse().unwrap();
            let size: u64 = parts[2].parse().unwrap();
            let ndim: usize = parts[3].parse().unwrap();
            let shape: Vec<u32> = parts[4..4 + ndim].iter().map(|s| s.parse().unwrap()).collect();
            let aligned = (size + 255) & !255;
            if cur.is_none() || used + aligned > chunk_size {
                if let Some(c) = cur.take() {
                    chunks.push(c);
                }
                cur = Some(gpu.buffer(chunk_size, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC));
                used = 0;
            }
            let mut data = vec![0u8; size as usize];
            file.seek(SeekFrom::Start(offset)).unwrap();
            file.read_exact(&mut data).unwrap();
            let chunk = cur.as_ref().unwrap();
            gpu.queue.write_buffer(&chunk.buffer, used, &data);
            entries.insert(name, (chunks.len(), used, size, shape));
            used += aligned;
            total += size;
            count += 1;
            if count % 300 == 0 {
                println!("  weights {count} {:.2} GB", total as f64 / 1e9);
            }
        }
        if let Some(c) = cur.take() {
            chunks.push(c);
        }
        println!(
            "weights loaded: {:.2} GB, {} tensors, {} chunks, {:.1}s",
            total as f64 / 1e9,
            count,
            chunks.len(),
            t0.elapsed().as_secs_f64()
        );
        Self { entries, chunks }
    }

    pub fn get(&self, name: &str) -> Buf {
        let (chunk, off, size, _) = self.entries.get(name).unwrap_or_else(|| panic!("missing weight {name}"));
        Buf { buffer: self.chunks[*chunk].buffer.clone(), offset: *off, size: *size }
    }

    pub fn try_get(&self, name: &str) -> Option<Buf> {
        self.entries.get(name).map(|(chunk, off, size, _)| Buf {
            buffer: self.chunks[*chunk].buffer.clone(),
            offset: *off,
            size: *size,
        })
    }

    pub fn shape(&self, name: &str) -> Vec<u32> {
        self.entries.get(name).unwrap_or_else(|| panic!("missing weight {name}")).3.clone()
    }
}
