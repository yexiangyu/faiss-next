use std::time::Instant;

use faiss_next::pairwise_l2_sqr;
use ndarray::Array2;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

fn pairwise_l2_ndarray(x: &Array2<f32>, y: &Array2<f32>) -> Array2<f32> {
    let nq = x.nrows();
    let nb = y.nrows();
    let d = x.ncols();

    let mut dist = Array2::<f32>::zeros((nq, nb));

    for i in 0..nq {
        let xi = x.row(i);
        for j in 0..nb {
            let yj = y.row(j);
            let mut sum = 0.0f32;
            for k in 0..d {
                let diff = xi[k] - yj[k];
                sum += diff * diff;
            }
            dist[[i, j]] = sum;
        }
    }

    dist
}

fn pairwise_l2_ndarray_broadcast(x: &Array2<f32>, y: &Array2<f32>) -> Array2<f32> {
    let nq = x.nrows();

    let mut dist = Array2::<f32>::zeros((nq, y.nrows()));

    for i in 0..nq {
        let xi = x.row(i);
        let xi_owned = xi.to_owned();
        for j in 0..y.nrows() {
            let yj = y.row(j);
            let diff = &xi_owned - &yj;
            dist[[i, j]] = diff.dot(&diff);
        }
    }

    dist
}

fn generate_data(n: usize, d: usize) -> (Array2<f32>, Vec<f32>) {
    let arr = Array2::random((n, d), Uniform::new(-1.0f32, 1.0f32));
    let flat: Vec<f32> = arr.iter().copied().collect();
    (arr, flat)
}

fn benchmark(_name: &str, iterations: usize, f: impl Fn()) -> f64 {
    let start = Instant::now();
    for _ in 0..iterations {
        f();
    }
    let elapsed = start.elapsed().as_secs_f64();
    elapsed / iterations as f64
}

fn main() {
    println!("L2 Distance Benchmark: ndarray vs faiss");
    println!("==========================================\n");

    let dimensions = [32, 64, 128, 256];
    let sizes = [(100, 1000), (1000, 1000), (1000, 10000)];
    let iterations = 3;

    for &d in &dimensions {
        println!("Dimension: {}", d);
        println!("{}", "-".repeat(60));
        println!(
            "{:15} {:15} {:15} {:>10}",
            "Size (nq, nb)", "ndarray (ms)", "faiss (ms)", "Speedup"
        );
        println!("{}", "-".repeat(60));

        for (nq, nb) in sizes {
            let (x_arr, x_flat) = generate_data(nq, d);
            let (y_arr, y_flat) = generate_data(nb, d);

            let t_ndarray = benchmark("ndarray", iterations, || {
                let _ = pairwise_l2_ndarray(&x_arr, &y_arr);
            });

            let t_faiss = benchmark("faiss", iterations, || {
                let _ = pairwise_l2_sqr(d, &x_flat, &y_flat);
            });

            let speedup = t_ndarray / t_faiss;

            println!(
                "({:>5}, {:>5}) {:>12.2} {:>12.2} {:>10.1}x",
                nq,
                nb,
                t_ndarray * 1000.0,
                t_faiss * 1000.0,
                speedup
            );
        }
        println!();
    }

    println!("\n=== Large Scale Test (Faiss only) ===\n");

    let d = 128usize;
    let large_sizes = [(10000, 10000), (10000, 50000), (50000, 50000)];

    println!("Dimension: {}", d);
    println!("{}", "-".repeat(50));
    println!("{:20} {:>15}", "Size (nq, nb)", "faiss (ms)");
    println!("{}", "-".repeat(50));

    for (nq, nb) in large_sizes {
        let (_, x_flat) = generate_data(nq, d);
        let (_, y_flat) = generate_data(nb, d);

        let t_faiss = benchmark("faiss", 1, || {
            let _ = pairwise_l2_sqr(d, &x_flat, &y_flat);
        });

        println!("({:>6}, {:>6}) {:>15.2}", nq, nb, t_faiss * 1000.0);
    }

    println!("\n=== ndarray dot product vs naive loop ===\n");

    let d = 64usize;
    let (x_arr, _) = generate_data(100, d);
    let (y_arr, _) = generate_data(1000, d);

    let t_naive = benchmark("naive", iterations, || {
        let _ = pairwise_l2_ndarray(&x_arr, &y_arr);
    });

    let t_dot = benchmark("dot", iterations, || {
        let _ = pairwise_l2_ndarray_broadcast(&x_arr, &y_arr);
    });

    println!("Dimension: {}, Size: (100, 1000)", d);
    println!("Naive loop:    {:.2} ms", t_naive * 1000.0);
    println!("Dot product:   {:.2} ms", t_dot * 1000.0);
    println!("Dot overhead:  {:.1}x slower", t_dot / t_naive);
}
