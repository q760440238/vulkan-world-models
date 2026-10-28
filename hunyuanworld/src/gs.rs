#[derive(Clone, Debug)]
pub struct Splats {
    pub means: Vec<[f32; 3]>,
    pub quats: Vec<[f32; 4]>,
    pub scales: Vec<[f32; 3]>,
    pub opacities: Vec<f32>,
    pub sh: Vec<[f32; 3]>,
}

pub struct SplatView<'a> {
    pub raw: &'a [f32],
    pub depth: &'a [f32],
    pub c2w: &'a [f32; 16],
    pub intr: &'a [f32; 9],
    pub image: &'a [f32],
}

fn sigmoid(x: f32) -> f32 { 1.0 / (1.0 + (-x).exp()) }

#[allow(clippy::too_many_arguments)]
pub fn build_splats(
    raw: &[f32],
    depth: &[f32],
    c2w: &[f32; 16],
    intr: &[f32; 9],
    image: &[f32],
    h: usize,
    w: usize,
    voxel_size: f32,
) -> Splats {
    let view = SplatView { raw, depth, c2w, intr, image };
    build_splats_multi(&[view], h, w, voxel_size)
}

pub fn build_splats_multi(views: &[SplatView<'_>], h: usize, w: usize, voxel_size: f32) -> Splats {
    let hw = h * w;
    let total = hw * views.len();
    let mut means = Vec::with_capacity(total);
    let mut quats = Vec::with_capacity(total);
    let mut scales = Vec::with_capacity(total);
    let mut opacities = Vec::with_capacity(total);
    let mut shs = Vec::with_capacity(total);
    let mut weights = Vec::with_capacity(total);
    let c0 = 0.28209479177387814f32;

    for view in views {
        assert_eq!(view.raw.len(), 12 * hw);
        assert_eq!(view.depth.len(), hw);
        assert_eq!(view.image.len(), 3 * hw);
        let fx = view.intr[0];
        let fy = view.intr[4];
        let cx = view.intr[2];
        let cy = view.intr[5];
        for p in 0..hw {
            let u = (p % w) as f32;
            let v = (p / w) as f32;
            let z = view.depth[p];
            let xc = (u - cx) * z / fx;
            let yc = (v - cy) * z / fy;
            let c2w = view.c2w;
            let m = [
                c2w[0] * xc + c2w[1] * yc + c2w[2] * z + c2w[3],
                c2w[4] * xc + c2w[5] * yc + c2w[6] * z + c2w[7],
                c2w[8] * xc + c2w[9] * yc + c2w[10] * z + c2w[11],
            ];
            let raw = view.raw;
            let q = [raw[p], raw[hw + p], raw[2 * hw + p], raw[3 * hw + p]];
            let qn = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
            let q = [q[0] / (qn + 1e-8), q[1] / (qn + 1e-8), q[2] / (qn + 1e-8), q[3] / (qn + 1e-8)];
            let s = [
                raw[4 * hw + p].exp().min(0.3),
                raw[5 * hw + p].exp().min(0.3),
                raw[6 * hw + p].exp().min(0.3),
            ];
            let o = sigmoid(raw[7 * hw + p]);
            let sh = [
                raw[8 * hw + p] + (view.image[p] - 0.5) / c0,
                raw[9 * hw + p] + (view.image[hw + p] - 0.5) / c0,
                raw[10 * hw + p] + (view.image[2 * hw + p] - 0.5) / c0,
            ];
            let wt = sigmoid(raw[11 * hw + p]);
            means.push(m);
            quats.push(q);
            scales.push(s);
            opacities.push(o);
            shs.push(sh);
            weights.push(wt);
        }
    }

    // Voxel pruning, matching torch.unique(sorted) + weighted scatter-add.
    let mut vox = Vec::with_capacity(total);
    let mut mins = [i64::MAX; 3];
    for m in &means {
        let v = [
            (m[0] / voxel_size).floor() as i64,
            (m[1] / voxel_size).floor() as i64,
            (m[2] / voxel_size).floor() as i64,
        ];
        for k in 0..3 { mins[k] = mins[k].min(v[k]); }
        vox.push(v);
    }
    let mut maxs = [i64::MIN; 3];
    for v in &vox {
        for k in 0..3 { maxs[k] = maxs[k].max(v[k] - mins[k]); }
    }
    let d1 = (maxs[1] + 1) as u64;
    let d2 = (maxs[2] + 1) as u64;
    let mut flat: Vec<(u64, usize)> = vox
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let x = (v[0] - mins[0]) as u64;
            let y = (v[1] - mins[1]) as u64;
            let z = (v[2] - mins[2]) as u64;
            (x * d1 * d2 + y * d2 + z, i)
        })
        .collect();
    flat.sort_by_key(|x| x.0);
    let mut out = Splats { means: vec![], quats: vec![], scales: vec![], opacities: vec![], sh: vec![] };
    let mut i = 0usize;
    while i < flat.len() {
        let key = flat[i].0;
        let mut j = i;
        while j < flat.len() && flat[j].0 == key { j += 1; }
        let mut wsum = 0.0f32;
        let mut mean = [0.0f32; 3];
        let mut sh = [0.0f32; 3];
        let mut sc = [0.0f32; 3];
        let mut quat = [0.0f32; 4];
        let mut op = 0.0f32;
        for (_, idx) in &flat[i..j] {
            let wt = weights[*idx];
            wsum += wt;
            for k in 0..3 { mean[k] += means[*idx][k] * wt; sh[k] += shs[*idx][k] * wt; sc[k] += scales[*idx][k] * wt; }
            for k in 0..4 { quat[k] += quats[*idx][k] * wt; }
            op += wt * wt;
        }
        let ws = wsum.max(1e-8);
        for k in 0..3 { mean[k] /= ws; sh[k] /= ws; sc[k] /= ws; }
        op /= ws;
        let qn = (quat[0] * quat[0] + quat[1] * quat[1] + quat[2] * quat[2] + quat[3] * quat[3]).sqrt().max(1e-8);
        for k in 0..4 { quat[k] /= qn; }
        out.means.push(mean);
        out.sh.push(sh);
        out.scales.push(sc);
        out.quats.push(quat);
        out.opacities.push(op);
        i = j;
    }
    out
}

pub fn flatten(s: &Splats) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut means = Vec::with_capacity(s.means.len() * 3);
    let mut quats = Vec::with_capacity(s.quats.len() * 4);
    let mut scales = Vec::with_capacity(s.scales.len() * 3);
    let mut sh = Vec::with_capacity(s.sh.len() * 3);
    for i in 0..s.means.len() {
        means.extend_from_slice(&s.means[i]);
        quats.extend_from_slice(&s.quats[i]);
        scales.extend_from_slice(&s.scales[i]);
        sh.extend_from_slice(&s.sh[i]);
    }
    (means, quats, scales, s.opacities.clone(), sh)
}
