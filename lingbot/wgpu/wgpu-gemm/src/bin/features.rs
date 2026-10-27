fn main() {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor { backends: wgpu::Backends::VULKAN, ..Default::default() });
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None, force_fallback_adapter: false,
    })).expect("adapter");
    let f = adapter.features();
    println!("adapter: {:?}", adapter.get_info());
    println!("SHADER_F16={}", f.contains(wgpu::Features::SHADER_F16));
    println!("FLOAT32_FILTERABLE={}", f.contains(wgpu::Features::FLOAT32_FILTERABLE));
    println!("features={:?}", f);
}
