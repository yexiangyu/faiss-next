use faiss::index::common::FaissIndexTrait;
use faiss::index::id_selector::FaissIDSelectorTrait;
use faiss_next as faiss;
use ndarray::Array2;
use ndarray_rand::RandomExt;
use tracing::*;

fn main() -> faiss::error::Result<()> {
    std::env::set_var("RUST_LOG", "trace");

    tracing_subscriber::fmt::init();

    let mut index = faiss::index::factory::faiss_index_factory(
        "Flat",
        128,
        faiss::index::metric::FaissMetricType::METRIC_L2,
    )?;

    let base = Array2::random(
        [1024, 128],
        ndarray_rand::rand_distr::Uniform::new(0.0, 1.0),
    );

    index.add(base.as_slice_memory_order().unwrap())?;

    let query = base.slice(ndarray::s![42..43, ..]);

    let (labels, distances) = index.search(query.as_slice_memory_order().unwrap(), 1)?;

    info!(?labels, ?distances, "search without params");

    let id_selector = faiss::index::id_selector::FaissIDSelectorRange::new(100, 1024)?.not()?;

    let params = faiss::index::parameters::FaissSearchParametersImpl::new(id_selector)?;

    let (labels, distances) =
        index.search_with_params(query.as_slice_memory_order().unwrap(), 1, &params)?;
    info!(?labels, ?distances, "search with params");

    let result = index.range_search(query.as_slice_memory_order().unwrap(), 1.00)?;

    let lims = result.lims();
    info!(?lims, "range search");

    let labels = result.labels();

    info!(?labels, "range search");
    Ok(())
}
