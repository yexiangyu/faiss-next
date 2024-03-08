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

    let query = base.slice(ndarray::s![42..43, ..]);

    let query = query.as_slice_memory_order().unwrap();
    let base = base.as_slice_memory_order().unwrap();

    let base_index = FaissIndexFlatImpl::new(128, FaissMetricType::METRIC_L2)?;
    let mut index = FaissIndexRefineFlatImpl::new(base_index)?;

    index.add(base)?;

    let (labels, distances) = index.search(query, 1)?;

    info!(?labels, ?distances);

    Ok(())
}
