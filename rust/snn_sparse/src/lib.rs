//! BioSNN 稀疏 LIF 前向核 —— 最小可测实现。
//!
//! **只做前向**，口径与 `research/sel/lm/model.py::TokenEProp.step` 一致：
//!
//! ```text
//! new_v  = alpha * v + current - z * threshold
//! scaled = (new_v - threshold) / threshold
//! z_new  = (scaled > 0) && (refractory <= 0)
//! r_new  = clamp(r + n_refractory * z_new - 1, 0, n_refractory)
//! ```
//!
//! 判据 R-0（与 PyTorch 参考差 <= 1e-5）与 R-1（稀疏 vs 稠密 >= 5x）见立项文档。

/// 单个 LIF 层的常数项。
#[derive(Clone, Copy, Debug)]
pub struct Lif {
    pub threshold: f32,
    pub alpha: f32,
    pub n_refractory: f32,
}

/// 与 `alpha` 同样口径：`exp(-1/tau)`。
pub fn alpha_from_tau(tau: f32) -> f32 {
    (-1.0 / tau).exp()
}

impl Lif {
    pub fn new(tau: f32, threshold: f32, n_refractory: u32) -> Self {
        Self { threshold, alpha: alpha_from_tau(tau), n_refractory: n_refractory as f32 }
    }

    /// 单步稠密更新：`current` / `v` / `z` / `r` 长度相同。
    pub fn step(&self, v: &mut [f32], z: &mut [f32], r: &mut [f32], current: &[f32]) {
        debug_assert_eq!(v.len(), current.len());
        for i in 0..v.len() {
            let z_prev = z[i];
            let new_v = self.alpha * v[i] + current[i] - z_prev * self.threshold;
            let scaled = (new_v - self.threshold) / self.threshold;
            let fires = scaled > 0.0 && r[i] <= 0.0;
            let new_z = if fires { 1.0 } else { 0.0 };
            r[i] = (r[i] + self.n_refractory * new_z - 1.0).clamp(0.0, self.n_refractory);
            v[i] = new_v;
            z[i] = new_z;
        }
    }
}

/// **稀疏读出**：只在 `idx` 指向的行上累加，`w` 是 `(n_rows, n_out)` 行主序。
///
/// 这是本项目要验证的那一刀：稠密版要把 `n_rows * n_out` 全读一遍，
/// 稀疏版只读 `idx.len() * n_out`。
pub fn accumulate_rows(w: &[f32], n_out: usize, idx: &[usize], vals: &[f32], out: &mut [f32]) {
    debug_assert_eq!(out.len(), n_out);
    debug_assert_eq!(idx.len(), vals.len());
    for (k, &row) in idx.iter().enumerate() {
        let base = row * n_out;
        let v = vals[k];
        let src = &w[base..base + n_out];
        for j in 0..n_out {
            out[j] += src[j] * v;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lif_matches_hand_computation() {
        let lif = Lif::new(20.0, 0.62, 2);
        let mut v = vec![0.0f32];
        let mut z = vec![0.0f32];
        let mut r = vec![0.0f32];
        lif.step(&mut v, &mut z, &mut r, &[0.9]);
        assert!((v[0] - 0.9).abs() < 1e-6, "v={}", v[0]);
        assert_eq!(z[0], 1.0, "0.9 > 0.62 应发放");
        assert!((r[0] - 1.0).abs() < 1e-6, "不应期应变为 1");
        // 下一步：r > 0 ⇒ 不发放
        lif.step(&mut v, &mut z, &mut r, &[0.9]);
        assert_eq!(z[0], 0.0, "不应期内不该发放");
    }

    #[test]
    fn sparse_accumulate_matches_dense() {
        let n_rows = 8usize;
        let n_out = 4usize;
        let w: Vec<f32> = (0..n_rows * n_out).map(|i| i as f32 * 0.5).collect();
        let vals: Vec<f32> = vec![1.0, 2.0];
        let idx: Vec<usize> = vec![1, 5];

        let mut sparse = vec![0.0f32; n_out];
        accumulate_rows(&w, n_out, &idx, &vals, &mut sparse);

        let mut dense = vec![0.0f32; n_out];
        for row in 0..n_rows {
            let v = if idx.contains(&row) { vals[idx.iter().position(|&x| x == row).unwrap()] } else { 0.0 };
            for j in 0..n_out {
                dense[j] += w[row * n_out + j] * v;
            }
        }
        for j in 0..n_out {
            assert!((sparse[j] - dense[j]).abs() < 1e-5);
        }
    }
}
