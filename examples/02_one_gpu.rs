use std::time::Instant;

use faiss::index::SearchParameters;
use faiss_next as faiss;

use faiss::error::{Error, Result};
use faiss::prelude::*;
use ndarray::Array2;
use ndarray_rand::{rand_distr::Uniform, RandomExt};
use tracing::*;

#[cfg(feature = "gpu")]
use faiss::gpu::cloner_options::GpuClonerOptions;
#[cfg(feature = "gpu")]
use faiss::gpu::index::IndexGpuImpl;
#[cfg(feature = "gpu")]
use faiss::gpu::standard_resources::StandardGpuResources;

#[cfg(not(feature = "gpu"))]
fn main() -> Result<()> {
    // init logger
    if std::env::var("RUST_LOG").is_err() {
        std::env::set_var("RUST_LOG", "trace");
    }
    tracing_subscriber::fmt::init();
    error!("this example requires `gpu` feature enabled");
    Ok(())
}

#[cfg(feature = "gpu")]
fn main() -> Result<()> {
    // init logger
    if std::env::var("RUST_LOG").is_err() {
        std::env::set_var("RUST_LOG", "trace");
    }
    tracing_subscriber::fmt::init();

    // dimension of vector
    let d = 64;

    // num of vector in base
    let n = 1024;

    // top k nearest neighbors
    let k = 1;

    // query id in base
    let id = 42;

    // initialize base and query
    let base = Array2::<f32>::random([n, d], Uniform::new(-1.0, 1.0));
    let base = base
        .as_slice_memory_order()
        .ok_or(Error::NotStandardLayout)?;

    let query = &base[id * d..(id + 1) * d];

    // convert index to gpu
    let mut index = {
        // create index by index_factory
        let index = faiss::index::factory::index_factory(d, "Flat", MetricType::L2)?;
        IndexGpuImpl::new(
            vec![StandardGpuResources::new()?],
            vec![0],
            &index,
            Option::<GpuClonerOptions>::None,
        )?
    };

    // add base vectors into index, without id;
    index.add(base, Option::<&[i64]>::None)?;

    // create distance and labels to store search result
    let mut distances = vec![0.0f32; k];
    let mut labels = vec![0i64; k];

    // do search
    let tm = Instant::now();

    index.search(
        query,
        k,
        &mut distances,
        &mut labels,
        Option::<SearchParameters>::None,
    )?;

    info!(?labels, ?distances, "search delta={:?}", tm.elapsed());
    Ok(())
}
