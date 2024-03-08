use faiss::error::Result;
use faiss::index::factory::FaissIndexBorrowed;
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

    let query = base.slice(ndarray::s![42..43, ..]);

    let query = query.as_slice_memory_order().unwrap();
    let base = base.as_slice_memory_order().unwrap();

    let i1 = FaissIndexLSH::new(128, 1024)?;
    let i2 = FaissIndexLSH::new(128, 1024)?;

    let mut index = FaissIndexReplicas::new_with_options(128, true)?;

    let i1_ = i1.inner();
    let i1_ = FaissIndexBorrowed::from(i1_);

    let i2_ = i2.inner();
    let i2_ = FaissIndexBorrowed::from(i2_);

    index.add_replica(i1)?;
    index.add_replica(i2)?;

    info!(
        "index d={}, ntotal={}, is_trained={}",
        index.d(),
        index.ntotal(),
        index.is_trained(),
    );

    index.add(base)?;

    info!(
        "index d={}, ntotal={}, is_trained={}, i1.ntotal={}, i2.ntotal={}",
        index.d(),
        index.ntotal(),
        index.is_trained(),
        i1_.ntotal(),
        i2_.ntotal()
    );

    let (labels, distances) = index.search(query, 1)?;

    info!(?labels, ?distances);

    Ok(())
}
