use crate::gs::Splats;

fn norm(v: [f32; 3]) -> [f32; 3] {
    let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-12);
    [v[0] / n, v[1] / n, v[2] / n]
}
fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

pub fn look_at(eye: [f32; 3], target: [f32; 3]) -> [f32; 16] {
    let z = norm([target[0] - eye[0], target[1] - eye[1], target[2] - eye[2]]);
    let x = norm(cross([0.0, 1.0, 0.0], z));
    let y = cross(z, x);
    [
        x[0], y[0], z[0], eye[0],
        x[1], y[1], z[1], eye[1],
        x[2], y[2], z[2], eye[2],
        0.0, 0.0, 0.0, 1.0,
    ]
}

/// Render one view with isotropic Gaussian splats.
///
/// The model's 3DGS output is a standard PLY; the learned anisotropic
/// rotations are preserved in the file, but this lightweight viewer uses an
/// isotropic projection for robustness and speed. Use gsplat/VkSplat for
/// anisotropic production rendering.
pub fn render_view(splats: &Splats, c2w: &[f32; 16], intr: &[f32; 9], size: usize, bg: f32) -> Vec<u8> {
    let s = size as f32 / 518.0;
    let fx = intr[0] * s;
    let fy = intr[4] * s;
    let cx = intr[2] * s;
    let cy = intr[5] * s;
    let mut order: Vec<usize> = (0..splats.means.len()).collect();
    let mut cam = vec![[0.0f32; 3]; splats.means.len()];
    let r = c2w;
    for i in 0..splats.means.len() {
        let p = splats.means[i];
        let x = p[0] - r[3];
        let y = p[1] - r[7];
        let z = p[2] - r[11];
        cam[i] = [
            r[0] * x + r[4] * y + r[8] * z,
            r[1] * x + r[5] * y + r[9] * z,
            r[2] * x + r[6] * y + r[10] * z,
        ];
    }
    order.sort_by(|&a, &b| cam[b][2].partial_cmp(&cam[a][2]).unwrap_or(std::cmp::Ordering::Equal));
    let mut img = vec![bg; size * size * 3];
    for &i in &order {
        let z = cam[i][2];
        if z <= 0.05 { continue; }
        let u = fx * cam[i][0] / z + cx;
        let v = fy * cam[i][1] / z + cy;
        let sc = splats.scales[i][0].max(splats.scales[i][1]).max(splats.scales[i][2]);
        let var = (sc * fx / z) * (sc * fx / z) + 0.3;
        let rad = (3.0 * var.sqrt()).ceil() as i32 + 1;
        if rad > 96 { continue; }
        let x0 = (u as i32 - rad).max(0) as usize;
        let x1 = ((u as i32 + rad + 1).max(0) as usize).min(size);
        let y0 = (v as i32 - rad).max(0) as usize;
        let y1 = ((v as i32 + rad + 1).max(0) as usize).min(size);
        if x0 >= x1 || y0 >= y1 { continue; }
        let c0 = 0.28209479177387814f32;
        let col = [
            (splats.sh[i][0] * c0 + 0.5).clamp(0.0, 1.0),
            (splats.sh[i][1] * c0 + 0.5).clamp(0.0, 1.0),
            (splats.sh[i][2] * c0 + 0.5).clamp(0.0, 1.0),
        ];
        let op = splats.opacities[i].clamp(0.0, 0.995);
        for yy in y0..y1 {
            for xx in x0..x1 {
                let dx = xx as f32 + 0.5 - u;
                let dy = yy as f32 + 0.5 - v;
                let a = op * (-0.5 * (dx * dx + dy * dy) / var).exp();
                if a < 0.003 { continue; }
                let k = (yy * size + xx) * 3;
                for ch in 0..3 {
                    img[k + ch] = a * col[ch] + (1.0 - a) * img[k + ch];
                }
            }
        }
    }
    img.iter().map(|v| (v.clamp(0.0, 1.0) * 255.0) as u8).collect()
}

pub fn orbit_camera(splats: &Splats, base: &[f32; 16], angle: f32) -> [f32; 16] {
    let mut lo = [f32::INFINITY; 3];
    let mut hi = [f32::NEG_INFINITY; 3];
    for m in &splats.means {
        for k in 0..3 { lo[k] = lo[k].min(m[k]); hi[k] = hi[k].max(m[k]); }
    }
    let center = [(lo[0] + hi[0]) * 0.5, (lo[1] + hi[1]) * 0.5, (lo[2] + hi[2]) * 0.5];
    let eye0 = [base[3], base[7], base[11]];
    let d = [eye0[0] - center[0], eye0[1] - center[1], eye0[2] - center[2]];
    let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt().max(1e-4);
    let (sa, ca) = angle.sin_cos();
    let eye = [center[0] + r * sa, eye0[1], center[2] + r * ca];
    look_at(eye, center)
}
