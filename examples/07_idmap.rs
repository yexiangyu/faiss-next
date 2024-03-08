use faiss::error::Result;
use faiss::prelude::*;
use faiss_next as faiss;
use ndarray_rand::RandomExt;
use tracing::*;

fn main() -> Result<()> {
    std::env::set_var("RUST_LOG", "trace");
    tracing_subscriber::fmt::init();

    let base = ndarray::Array2::random(
        [1024, 128],
        ndarray_rand::rand_distr::Uniform::new(0.0, 1.0),
    );
    let ids = ndarray::Array1::random(
        [1024],
        ndarray_rand::rand_distr::Uniform::new(0i64, 9999999999999999i64),
    );

    let query = base.slice(ndarray::s![42..43, ..]);

    let query = query.as_slice_memory_order().unwrap();
    let base = base.as_slice_memory_order().unwrap();
    let ids = ids.as_slice_memory_order().unwrap();

    let index = FaissIndexLSH::new(128, 1024)?;
    let mut index = FaissIndexIDMap::new(index)?;

    info!(
        "index d={}, ntotal={}, is_trained={}",
        index.d(),
        index.ntotal(),
        index.is_trained()
    );

    index.train(base)?;
    index.add_with_ids(base, ids)?;

    info!(
        "index d={}, ntotal={}, is_trained={}",
        index.d(),
        index.ntotal(),
        index.is_trained()
    );

    // index.train(base)?;

    // info!(
    //     "index d={}, ntotal={}, is_trained={}",
    //     index.d(),
    //     index.ntotal(),
    //     index.is_trained()
    // );

    let (labels, distances) = index.search(query, 1)?;

    info!(?labels, ?distances);
    info!("labels={}", ids[42]);

    Ok(())
}
