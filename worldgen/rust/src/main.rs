use image::imageops::FilterType;
use std::env;
use std::error::Error;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

const SH_C0: f32 = 0.282_094_79;

fn unit(v: [f32; 3]) -> [f32; 3] {
    let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if n < 1e-12 {
        [0.0, 0.0, 0.0]
    } else {
        [v[0] / n, v[1] / n, v[2] / n]
    }
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Quaternion (w, x, y, z) from a rotation matrix whose columns are the
/// local x/y/z axes in world space, matching pytorch3d's convention.
fn quat_from_rot(r: [[f32; 3]; 3]) -> [f32; 4] {
    let trace = r[0][0] + r[1][1] + r[2][2];
    let (w, x, y, z);
    if trace > 0.0 {
        let s = (trace + 1.0).sqrt() * 2.0;
        w = 0.25 * s;
        x = (r[2][1] - r[1][2]) / s;
        y = (r[0][2] - r[2][0]) / s;
        z = (r[1][0] - r[0][1]) / s;
    } else if r[0][0] > r[1][1] && r[0][0] > r[2][2] {
        let s = (1.0 + r[0][0] - r[1][1] - r[2][2]).sqrt() * 2.0;
        w = (r[2][1] - r[1][2]) / s;
        x = 0.25 * s;
        y = (r[0][1] + r[1][0]) / s;
        z = (r[0][2] + r[2][0]) / s;
    } else if r[1][1] > r[2][2] {
        let s = (1.0 + r[1][1] - r[0][0] - r[2][2]).sqrt() * 2.0;
        w = (r[0][2] - r[2][0]) / s;
        x = (r[0][1] + r[1][0]) / s;
        y = 0.25 * s;
        z = (r[1][2] + r[2][1]) / s;
    } else {
        let s = (1.0 + r[2][2] - r[0][0] - r[1][1]).sqrt() * 2.0;
        w = (r[1][0] - r[0][1]) / s;
        x = (r[0][2] + r[2][0]) / s;
        y = (r[1][2] + r[2][1]) / s;
        z = 0.25 * s;
    }
    let q = [w, x, y, z];
    let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    [q[0] / n, q[1] / n, q[2] / n, q[3] / n]
}

fn load_depth(path: &Path) -> Result<(Vec<f32>, usize, usize)> {
    let bytes = std::fs::read(path)?;
    if bytes.len() % 4 != 0 {
        return Err(format!("depth size {} is not a multiple of 4", bytes.len()).into());
    }
    let vals: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    // The caller supplies --width/--height; a square fallback is kept for
    // quick tests. The WorldGen pipeline writes 1600x800.
    Ok((vals, 0, 0))
}

fn pano_rgb(path: &Path, w: usize, h: usize) -> Result<Vec<u8>> {
    let img = image::open(path)?.to_rgb8();
    let resized = image::imageops::resize(&img, w as u32, h as u32, FilterType::Triangle);
    Ok(resized.into_raw())
}

fn rays(w: usize, h: usize) -> Vec<[f32; 3]> {
    let mut out = Vec::with_capacity(w * h);
    for y in 0..h {
        let v = (y as f32 + 0.5) / h as f32;
        let theta = v * std::f32::consts::PI - std::f32::consts::FRAC_PI_2;
        let (st, ct) = theta.sin_cos();
        for x in 0..w {
            let u = (x as f32 + 0.5) / w as f32;
            let phi = u * std::f32::consts::TAU - std::f32::consts::PI;
            let (sp, cp) = phi.sin_cos();
            out.push([ct * sp, st, ct * cp]);
        }
    }
    out
}

struct Splat {
    centers: Vec<[f32; 3]>,
    colors: Vec<[f32; 3]>,
    scales: Vec<[f32; 3]>,
    quats: Vec<[f32; 4]>,
}

fn build_splats(rgb: &[u8], depth: &[f32], w: usize, h: usize) -> Splat {
    let delta_phi = std::f32::consts::TAU / w as f32;
    let delta_theta = std::f32::consts::PI / h as f32;
    let scale_factor = 0.65f32;
    let mut s = Splat {
        centers: Vec::with_capacity(w * h),
        colors: Vec::with_capacity(w * h),
        scales: Vec::with_capacity(w * h),
        quats: Vec::with_capacity(w * h),
    };
    let rs = rays(w, h);
    for i in 0..w * h {
        let d = depth[i];
        if !(d > 0.0) || !d.is_finite() {
            continue;
        }
        let ray = rs[i];
        let y = i / w;
        // WorldGen uses the polar angle 0..pi for the equirectangular
        // sin(theta) footprint correction, not the latitude -pi/2..pi/2.
        let theta = (y as f32 + 0.5) / h as f32 * std::f32::consts::PI;
        let sigma_x = d * delta_phi * theta.sin() * scale_factor;
        let sigma_y = d * delta_theta * scale_factor;
        let sigma_z = 1e-3f32;
        let up = [0.0, 1.0, 0.0];
        let mut x_axis = unit(cross(up, ray));
        if x_axis[0].is_nan() || x_axis[1].is_nan() || x_axis[2].is_nan() {
            x_axis = unit(cross([1.0, 0.0, 0.0], ray));
        }
        let y_axis = unit(cross(ray, x_axis));
        let r = [
            [x_axis[0], y_axis[0], ray[0]],
            [x_axis[1], y_axis[1], ray[1]],
            [x_axis[2], y_axis[2], ray[2]],
        ];
        s.centers.push([ray[0] * d, ray[1] * d, ray[2] * d]);
        s.colors.push([
            rgb[i * 3] as f32 / 255.0,
            rgb[i * 3 + 1] as f32 / 255.0,
            rgb[i * 3 + 2] as f32 / 255.0,
        ]);
        s.scales.push([sigma_x, sigma_y, sigma_z]);
        s.quats.push(quat_from_rot(r));
    }
    s
}

fn write_gaussians(path: &Path, s: &Splat) -> Result<()> {
    let mut f = BufWriter::new(File::create(path)?);
    writeln!(f, "ply")?;
    writeln!(f, "format binary_little_endian 1.0")?;
    writeln!(f, "element vertex {}", s.centers.len())?;
    for p in ["x", "y", "z", "nx", "ny", "nz"] {
        writeln!(f, "property float {p}")?;
    }
    for i in 0..3 {
        writeln!(f, "property float f_dc_{i}")?;
    }
    writeln!(f, "property float opacity")?;
    for i in 0..3 {
        writeln!(f, "property float scale_{i}")?;
    }
    for i in 0..4 {
        writeln!(f, "property float rot_{i}")?;
    }
    writeln!(f, "end_header")?;
    for i in 0..s.centers.len() {
        let c = s.centers[i];
        let col = s.colors[i];
        let sc = s.scales[i];
        let q = s.quats[i];
        let mut row = Vec::with_capacity(17 * 4);
        for v in [
            c[0], c[1], c[2], 0.0, 0.0, 0.0,
            (col[0] - 0.5) / SH_C0,
            (col[1] - 0.5) / SH_C0,
            (col[2] - 0.5) / SH_C0,
            1.0,
            sc[0].ln(), sc[1].ln(), sc[2].ln(),
            q[0], q[1], q[2], q[3],
        ] {
            row.extend_from_slice(&v.to_le_bytes());
        }
        f.write_all(&row)?;
    }
    Ok(())
}

fn median3(depth: &[f32], w: usize, h: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut v = [0.0f32; 9];
            let mut n = 0;
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    let yy = (y as i32 + dy).clamp(0, h as i32 - 1) as usize;
                    let xx = (x as i32 + dx).rem_euclid(w as i32) as usize;
                    let d = depth[yy * w + xx];
                    if d.is_finite() && d > 0.0 {
                        v[n] = d;
                        n += 1;
                    }
                }
            }
            if n == 0 {
                out[y * w + x] = depth[y * w + x];
            } else {
                v[..n].sort_by(|a, b| a.partial_cmp(b).unwrap());
                out[y * w + x] = v[n / 2];
            }
        }
    }
    out
}

fn write_mesh(path: &Path, rgb: &[u8], depth: &[f32], w: usize, h: usize) -> Result<usize> {
    let rs = rays(w, h);
    let depth = median3(depth, w, h);
    let mut index = vec![-1i64; w * h];
    let mut verts: Vec<([f32; 3], [u8; 3])> = Vec::new();
    for i in 0..w * h {
        if depth[i] > 0.0 && depth[i].is_finite() {
            index[i] = verts.len() as i64;
            verts.push((
                [rs[i][0] * depth[i], rs[i][1] * depth[i], rs[i][2] * depth[i]],
                [rgb[i * 3], rgb[i * 3 + 1], rgb[i * 3 + 2]],
            ));
        }
    }
    let mut tris: Vec<[i32; 3]> = Vec::new();
    for y in 0..h - 1 {
        for x in 0..w {
            let x1 = (x + 1) % w;
            let ia = y * w + x;
            let ib = y * w + x1;
            let ic = (y + 1) * w + x;
            let id = (y + 1) * w + x1;
            let a = index[ia];
            let b = index[ib];
            let c = index[ic];
            let d = index[id];
            if a < 0 || b < 0 || c < 0 || d < 0 {
                continue;
            }
            let ds = [depth[ia], depth[ib], depth[ic], depth[id]];
            let dmin = ds.iter().cloned().fold(f32::INFINITY, f32::min);
            let dmax = ds.iter().cloned().fold(0.0f32, f32::max);
            // Do not bridge a foreground/background depth discontinuity with a
            // long, stretched quad; leave a hole at the silhouette instead.
            if dmin > 0.0 && dmax / dmin > 1.8 {
                continue;
            }
            tris.push([a as i32, c as i32, b as i32]);
            tris.push([b as i32, c as i32, d as i32]);
        }
    }
    let mut f = BufWriter::new(File::create(path)?);
    writeln!(f, "ply")?;
    writeln!(f, "format binary_little_endian 1.0")?;
    writeln!(f, "element vertex {}", verts.len())?;
    for p in ["x", "y", "z"] {
        writeln!(f, "property float {p}")?;
    }
    for p in ["red", "green", "blue"] {
        writeln!(f, "property uchar {p}")?;
    }
    writeln!(f, "element face {}", tris.len())?;
    writeln!(f, "property list uchar int vertex_indices")?;
    writeln!(f, "end_header")?;
    for (p, c) in &verts {
        for v in p {
            f.write_all(&v.to_le_bytes())?;
        }
        f.write_all(c)?;
    }
    for t in &tris {
        f.write_all(&[3u8])?;
        for v in t {
            f.write_all(&v.to_le_bytes())?;
        }
    }
    Ok(tris.len())
}

fn usage() -> ! {
    eprintln!(
        "worldgen-vk 0.1.0\n\
         usage:\n\
           worldgen-vk splat --pano P.png --depth D.f32 --width W --height H --out DIR\n\
           worldgen-vk mesh  --pano P.png --depth D.f32 --width W --height H --out DIR [--max-size N]\n\
           worldgen-vk both  --pano P.png --depth D.f32 --width W --height H --out DIR [--max-size N]\n\
         depth is raw little-endian f32, H*W, as written by the WorldGen/DA-2 bridge."
    );
    std::process::exit(2);
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        usage();
    }
    let mode = args[1].as_str();
    let mut pano = None;
    let mut depth_path = None;
    let mut out = None;
    let mut w = 0usize;
    let mut h = 0usize;
    let mut max_size = 4096usize;
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--pano" => { pano = Some(PathBuf::from(&args[i + 1])); i += 2; }
            "--depth" => { depth_path = Some(PathBuf::from(&args[i + 1])); i += 2; }
            "--out" => { out = Some(PathBuf::from(&args[i + 1])); i += 2; }
            "--width" => { w = args[i + 1].parse()?; i += 2; }
            "--height" => { h = args[i + 1].parse()?; i += 2; }
            "--max-size" => { max_size = args[i + 1].parse()?; i += 2; }
            _ => usage(),
        }
    }
    let (pano, depth_path, out) = match (pano, depth_path, out) {
        (Some(a), Some(b), Some(c)) => (a, b, c),
        _ => usage(),
    };
    if w == 0 || h == 0 {
        usage();
    }
    let (depth, _, _) = load_depth(&depth_path)?;
    if depth.len() != w * h {
        return Err(format!("depth has {} values, expected {}x{}", depth.len(), w, h).into());
    }
    std::fs::create_dir_all(&out)?;
    let rgb = pano_rgb(&pano, w, h)?;
    let mut made = Vec::new();
    let mut splat_count = 0;
    let mut tri_count = 0;
    if mode == "splat" || mode == "both" {
        let s = build_splats(&rgb, &depth, w, h);
        splat_count = s.centers.len();
        write_gaussians(&out.join("gaussians.ply"), &s)?;
        made.push("gaussians.ply");
    }
    if mode == "mesh" || mode == "both" {
        // WorldGen's torch mesh path caps the long edge at 4096.
        let scale = if w.max(h) > max_size {
            max_size as f32 / w.max(h) as f32
        } else {
            1.0
        };
        let (mw, mh) = (
            ((w as f32 * scale) as usize).max(2),
            ((h as f32 * scale) as usize).max(2),
        );
        let rgb2 = image::imageops::resize(
            &image::RgbImage::from_raw(w as u32, h as u32, rgb.clone()).unwrap(),
            mw as u32,
            mh as u32,
            FilterType::Triangle,
        )
        .into_raw();
        let mut depth2 = vec![0f32; mw * mh];
        for y in 0..mh {
            let sy = (y as f32 + 0.5) * h as f32 / mh as f32;
            for x in 0..mw {
                let sx = (x as f32 + 0.5) * w as f32 / mw as f32;
                let x0 = (sx as usize).min(w - 1);
                let y0 = (sy as usize).min(h - 1);
                depth2[y * mw + x] = depth[y0 * w + x0];
            }
        }
        tri_count = write_mesh(&out.join("mesh.ply"), &rgb2, &depth2, mw, mh)?;
        made.push("mesh.ply");
    }
    let meta = format!(
        "{{\n  \"mode\": \"{}\",\n  \"width\": {},\n  \"height\": {},\n  \"splats\": {},\n  \"triangles\": {},\n  \"files\": [{}],\n  \"note\": \"WorldGen RGBD panorama to 3DGS/mesh, Rust port\"\n}}\n",
        mode,
        w,
        h,
        splat_count,
        tri_count,
        made.iter().map(|s| format!("\"{s}\"")).collect::<Vec<_>>().join(", ")
    );
    std::fs::write(out.join("worldgen.json"), meta)?;
    println!(
        "{}: {} splats, {} triangles -> {}",
        mode,
        splat_count,
        tri_count,
        out.display()
    );
    Ok(())
}
