//! refer: https://github.com/facebookresearch/faiss/wiki/Faster-search

use std::time::Instant;

use faiss_next as faiss;
use ndarray_rand::RandomExt;
use rand::distributions::Uniform;

use faiss::index::Index;
use tracing::*;

pub fn main() -> faiss::error::Result<()> {
    if std::env::var("RUST_LOG").is_err() {
        std::env::set_var("RUST_LOG", "info");
    }
    tracing_subscriber::fmt::init();

    let d = 64i64;
    let n_list = 100;
    let nb = 100000usize;
    let nq = 32;
    let k = 4;

    let xb = ndarray::Array2::random([nb, d as usize], Uniform::new(0.0, 1.0));
    let xq = ndarray::Array2::random([nq, d as usize], Uniform::new(0.0, 1.0));

    let mut quantizer = faiss::index::flat::FaissIndexFlatL2::new_with(d)?;
    let mut index =
        faiss::index::flat::FaissIndexIVFFlat::new_with(&mut quantizer, d as usize, n_list)?;

    info!("index training");

    let tm = Instant::now();

    index.train(xb.as_slice_memory_order().expect("failed to use ndarray?"))?;

    info!("index trained={} in {:?}", index.is_trained(), tm.elapsed());

    index.add(xb.as_slice_memory_order().expect("failed to use ndarray?"))?;

    info!(
        "add: {:?} {} dimensions vetor into index",
        index.ntotal(),
        index.d()
    );

    let tm = Instant::now();

    let (labels, scores) = index.search(
        xq.as_slice_memory_order().expect("failed to use ndarray?"),
        k,
    )?;

    info!("search done in {:?}", tm.elapsed());

    let labels = ndarray::Array2::from_shape_vec([nq, k as usize], labels).expect("array?");
    let scores = ndarray::Array2::from_shape_vec([nq, k as usize], scores).expect("array?");

    for i in 0..nq {
        let label_i = labels
            .row(i)
            .as_slice_memory_order()
            .expect("?")
            .iter()
            .map(|v| format!("{:>5}", v))
            .collect::<Vec<_>>()
            .join("|");
        let score_i = scores
            .row(i)
            .as_slice_memory_order()
            .expect("?")
            .iter()
            .map(|v| format!("{:.2}", v))
            .collect::<Vec<_>>()
            .join("|");
        info!("i={:>2}, label=|{}|, score=|{}|", i, label_i, score_i);
    }

    Ok(())
}
