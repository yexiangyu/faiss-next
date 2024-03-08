use faiss::error::Result;
use faiss::index::pre_transform::FaissIndexPreTransformImpl;
use faiss::prelude::*;
use faiss::vector_transform::FaissRandomRotationMatrix;
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

    let index = FaissIndexFlatL2::new(64)?;

    let trans = FaissRandomRotationMatrix::new_with(128, 64)?;

    let mut index = FaissIndexPreTransformImpl::new_with_transform(trans, index)?;
    let mut index = FaissIndexPreTransformImpl::new_with(index)?;

    info!(
        "index d={}, ntotal={}, is_trained={}",
        index.d(),
        index.ntotal(),
        index.is_trained()
    );

    index.train(base)?;
    index.add(base)?;

    info!(
        "index d={}, ntotal={}, is_trained={}",
        index.d(),
        index.ntotal(),
        index.is_trained()
    );

    let (labels, distances) = index.search(query, 1)?;

    info!(?labels, ?distances);

    Ok(())
}
