//! R-1 基准：**稠密 vs 稀疏**读出的每 token 成本。
//!
//! 判据（`research/sel/plan-rust-sparse-kernel.md`）：密度 1%–4% 时 **>= 5x**。
//! 计时必须在不被打扰的机器上跑（建库会吃满 CPU）。

use std::time::Instant;

use snn_sparse::accumulate_rows;

fn bench_dense(w: &[f32], n_out: usize, vals: &[f32], out: &mut [f32], iters: usize) -> f64 {
    let started = Instant::now();
    for _ in 0..iters {
        out.iter_mut().for_each(|x| *x = 0.0);
        for (row, &v) in vals.iter().enumerate() {
            let base = row * n_out;
            for j in 0..n_out {
                out[j] += w[base + j] * v;
            }
        }
    }
    started.elapsed().as_secs_f64() / iters as f64
}

fn main() {
    // 当前工作点：M = 2*N = 16384，V = 3001（约 197 MB，与线上一致）
    let n_rows = 16_384usize;
    let n_out = 3_001usize;
    let w: Vec<f32> = (0..n_rows * n_out).map(|i| (i % 97) as f32 * 0.001).collect();
    let iters = 20;

    let mut out = vec![0.0f32; n_out];
    let dense_vals = vec![0.01f32; n_rows];
    let dense = bench_dense(&w, n_out, &dense_vals, &mut out, iters);

    println!("| 密度 | 稠密 (ms) | 稀疏 (ms) | 比 |");
    println!("| ---: | ---: | ---: | ---: |");
    for pct in [1usize, 2, 4, 10] {
        let active = n_rows * pct / 100;
        let idx: Vec<usize> = (0..active).map(|i| i * (n_rows / active)).collect();
        let vals = vec![0.01f32; active];
        let started = Instant::now();
        for _ in 0..iters {
            out.iter_mut().for_each(|x| *x = 0.0);
            accumulate_rows(&w, n_out, &idx, &vals, &mut out);
        }
        let sparse = started.elapsed().as_secs_f64() / iters as f64;
        println!(
            "| {:>2}% | {:.3} | {:.3} | **{:.1}x** |",
            pct,
            dense * 1e3,
            sparse * 1e3,
            dense / sparse
        );
    }
}
