struct Params {
    w: u32,
    h: u32,
    pad0: u32,
    pad1: u32,
};

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> rgb: array<u32>;
@group(0) @binding(2) var<storage, read> depth: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

const SH_C0: f32 = 0.28209479177387814;
const PI: f32 = 3.141592653589793;
const TAU: f32 = 6.283185307179586;

fn quat_from_rot(r00: f32, r01: f32, r02: f32,
                 r10: f32, r11: f32, r12: f32,
                 r20: f32, r21: f32, r22: f32) -> vec4<f32> {
    let tr = r00 + r11 + r22;
    var q: vec4<f32>;
    if (tr > 0.0) {
        let s = sqrt(tr + 1.0) * 2.0;
        q = vec4<f32>(0.25 * s, (r21 - r12) / s, (r02 - r20) / s, (r10 - r01) / s);
    } else if (r00 > r11 && r00 > r22) {
        let s = sqrt(1.0 + r00 - r11 - r22) * 2.0;
        q = vec4<f32>((r21 - r12) / s, 0.25 * s, (r01 + r10) / s, (r02 + r20) / s);
    } else if (r11 > r22) {
        let s = sqrt(1.0 + r11 - r00 - r22) * 2.0;
        q = vec4<f32>((r02 - r20) / s, (r01 + r10) / s, 0.25 * s, (r12 + r21) / s);
    } else {
        let s = sqrt(1.0 + r22 - r00 - r11) * 2.0;
        q = vec4<f32>((r10 - r01) / s, (r02 + r20) / s, (r12 + r21) / s, 0.25 * s);
    }
    return normalize(q);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let n = P.w * P.h;
    if (i >= n) { return; }
    let o = i * 17u;
    let x = i % P.w;
    let y = i / P.w;
    var d = depth[i];
    if (d <= 0.0 || !(d < 1e30)) {
        for (var k = 0u; k < 17u; k = k + 1u) { out[o + k] = 0.0; }
        return;
    }
    let u = (f32(x) + 0.5) / f32(P.w);
    let v = (f32(y) + 0.5) / f32(P.h);
    let phi = u * TAU - PI;
    let theta = v * PI - PI * 0.5;
    let st = sin(theta);
    let ct = cos(theta);
    let sp = sin(phi);
    let cp = cos(phi);
    let ray = vec3<f32>(ct * sp, st, ct * cp);
    let phi0 = (f32(y) + 0.5) / f32(P.h) * PI;
    let sf = 0.65;
    let sx = d * (TAU / f32(P.w)) * sin(phi0) * sf;
    let sy = d * (PI / f32(P.h)) * sf;
    let sz = 0.001;
    var xa = cross(vec3<f32>(0.0, 1.0, 0.0), ray);
    if (length(xa) < 1e-6) { xa = cross(vec3<f32>(1.0, 0.0, 0.0), ray); }
    xa = normalize(xa);
    let ya = normalize(cross(ray, xa));
    let q = quat_from_rot(xa.x, ya.x, ray.x, xa.y, ya.y, ray.y, xa.z, ya.z, ray.z);
    let c = rgb[i];
    let r = f32((c >> 0u) & 255u) / 255.0;
    let g = f32((c >> 8u) & 255u) / 255.0;
    let b = f32((c >> 16u) & 255u) / 255.0;
    out[o + 0u] = ray.x * d;
    out[o + 1u] = ray.y * d;
    out[o + 2u] = ray.z * d;
    out[o + 3u] = 0.0;
    out[o + 4u] = 0.0;
    out[o + 5u] = 0.0;
    out[o + 6u] = (r - 0.5) / SH_C0;
    out[o + 7u] = (g - 0.5) / SH_C0;
    out[o + 8u] = (b - 0.5) / SH_C0;
    out[o + 9u] = 1.0;
    out[o + 10u] = log(sx);
    out[o + 11u] = log(sy);
    out[o + 12u] = log(sz);
    out[o + 13u] = q.w;
    out[o + 14u] = q.x;
    out[o + 15u] = q.y;
    out[o + 16u] = q.z;
}
