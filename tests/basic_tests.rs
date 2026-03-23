use faiss_next::{
    index_factory, read_index, write_index, Clustering, ClusteringParameters, Index, IndexBuilder,
    IndexFlat, IndexIDMap, IndexIVF, IndexIVFFlat, IndexLSH, IvfIndex, MetricType,
    SearchParameters, SearchParametersIvf, SearchParams,
};
use std::path::Path;

fn generate_unique_data(n: usize, d: usize, seed: u64) -> Vec<f32> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    (0..n * d)
        .map(|i| {
            let mut hasher = DefaultHasher::new();
            (i as u64 * seed).hash(&mut hasher);
            let hash = hasher.finish();
            ((hash % 1000) as f32 / 100.0) - 5.0
        })
        .collect()
}

fn cleanup_test_file(path: &str) {
    let p = Path::new(path);
    if p.exists() {
        std::fs::remove_file(p).ok();
    }
}

#[test]
fn test_flat_index_l2() {
    let d = 64u32;
    let n = 100usize;
    let k = 10usize;

    let mut index = IndexFlat::new_l2(d).unwrap();
    let data = generate_unique_data(n, d as usize, 42);

    index.add(&data).unwrap();
    assert_eq!(index.ntotal(), n as u64);
    assert_eq!(index.d(), d);

    let query: Vec<f32> = data[0..d as usize].to_vec();
    let result = index.search(&query, k).unwrap();

    assert_eq!(result.labels.len(), k);
    assert_eq!(result.labels[0].get(), Some(0));
    assert!(result.distances[0] < 1e-5);
}

#[test]
fn test_flat_index_ip() {
    let d = 32u32;
    let n = 50usize;

    let mut index = IndexFlat::new_ip(d).unwrap();
    let data = generate_unique_data(n, d as usize, 123);

    index.add(&data).unwrap();
    assert_eq!(index.ntotal(), n as u64);

    let offset = d as usize * 10;
    let query: Vec<f32> = data[offset..offset + d as usize].to_vec();
    let result = index.search(&query, 5).unwrap();

    assert_eq!(result.labels[0].get(), Some(10));
}

#[test]
fn test_index_factory_flat() {
    let d = 32u32;
    let n = 100usize;

    let mut index = index_factory(d, "Flat", MetricType::L2).unwrap();
    let data = generate_unique_data(n, d as usize, 456);

    index.add(&data).unwrap();
    assert_eq!(index.ntotal(), n as u64);

    let offset = d as usize * 50;
    let query: Vec<f32> = data[offset..offset + d as usize].to_vec();
    let result = index.search(&query, 5).unwrap();

    assert_eq!(result.labels[0].get(), Some(50));
}

#[test]
fn test_index_factory_ivf_flat() {
    let d = 32u32;
    let n = 1000usize;

    let mut index = index_factory(d, "IVF10,Flat", MetricType::L2).unwrap();
    let data = generate_unique_data(n, d as usize, 789);

    index.train(&data).unwrap();
    index.add(&data).unwrap();
    assert_eq!(index.ntotal(), n as u64);

    let offset = d as usize * 100;
    let query: Vec<f32> = data[offset..offset + d as usize].to_vec();
    let result = index.search(&query, 5).unwrap();

    assert!(result.labels[0].is_some());
}

#[test]
fn test_index_builder() {
    let d = 16u32;
    let n = 200usize;

    let mut index = IndexBuilder::new(d).hnsw(32).l2().build().unwrap();
    let data = generate_unique_data(n, d as usize, 999);

    index.add(&data).unwrap();
    assert_eq!(index.ntotal(), n as u64);

    let offset = d as usize * 100;
    let query: Vec<f32> = data[offset..offset + d as usize].to_vec();
    let result = index.search(&query, 5).unwrap();

    assert_eq!(result.labels[0].get(), Some(100));
}

#[test]
fn test_concurrent_search() {
    let d = 32u32;
    let n = 100usize;

    let mut index = IndexFlat::new_l2(d).unwrap();
    let data = generate_unique_data(n, d as usize, 111);

    index.add(&data).unwrap();

    let query: Vec<f32> = data[0..d as usize].to_vec();
    let result = index.search(&query, 5).unwrap();

    assert_eq!(result.labels[0].get(), Some(0));
}

#[test]
fn test_index_reset() {
    let d = 16u32;
    let n = 50usize;

    let mut index = IndexFlat::new_l2(d).unwrap();
    let data = generate_unique_data(n, d as usize, 222);

    index.add(&data).unwrap();
    assert_eq!(index.ntotal(), n as u64);

    index.reset().unwrap();
    assert_eq!(index.ntotal(), 0);
}

#[test]
fn test_ivf_flat_index() {
    let d = 32u32;
    let nlist = 10usize;
    let n = 1000usize;

    let quantizer = IndexFlat::new_l2(d).unwrap();
    let mut index = IndexIVFFlat::new(quantizer, nlist).unwrap();

    assert_eq!(index.nlist(), nlist);

    let data = generate_unique_data(n, d as usize, 333);

    index.train(&data).unwrap();
    index.add(&data).unwrap();
    assert_eq!(index.ntotal(), n as u64);

    let offset = d as usize * 50;
    let query: Vec<f32> = data[offset..offset + d as usize].to_vec();
    let result = index.search(&query, 5).unwrap();

    assert!(result.labels[0].is_some());
}

#[test]
fn test_ivf_index_nprobe() {
    let d = 16u32;
    let nlist = 5usize;
    let n = 500usize;

    let index = index_factory(d, &format!("IVF{},Flat", nlist), MetricType::L2).unwrap();
    let ivf = IndexIVF::from_index(index).unwrap();

    let data = generate_unique_data(n, d as usize, 444);

    let mut ivf = ivf;
    ivf.train(&data).unwrap();
    ivf.add(&data).unwrap();

    let default_nprobe = ivf.nprobe();
    assert!(default_nprobe > 0);

    ivf.set_nprobe(3);
    assert_eq!(ivf.nprobe(), 3);
}

#[test]
fn test_ivf_index_from_factory() {
    let d = 32u32;
    let nlist = 10usize;
    let n = 1000usize;

    let index = index_factory(d, &format!("IVF{},Flat", nlist), MetricType::L2).unwrap();
    let data = generate_unique_data(n, d as usize, 555);

    let mut index = index;
    index.train(&data).unwrap();
    index.add(&data).unwrap();

    let ivf = IndexIVF::from_index(index).unwrap();
    assert_eq!(ivf.nlist(), nlist);
    assert_eq!(ivf.ntotal(), n as u64);
}

#[test]
fn test_lsh_index() {
    let d = 16u32;
    let nbits = 32u32;
    let n = 200usize;

    let mut index = IndexLSH::new(d, nbits).unwrap();
    assert_eq!(index.nbits(), nbits);

    let data = generate_unique_data(n, d as usize, 666);

    index.train(&data).unwrap();
    index.add(&data).unwrap();
    assert_eq!(index.ntotal(), n as u64);

    let offset = d as usize * 50;
    let query: Vec<f32> = data[offset..offset + d as usize].to_vec();
    let result = index.search(&query, 5).unwrap();

    assert!(result.labels[0].is_some());
}

#[test]
fn test_id_map() {
    let d = 16u32;
    let n = 100usize;

    let base_index = index_factory(d, "Flat", MetricType::L2).unwrap();
    let mut index = IndexIDMap::new(base_index).unwrap();

    let data = generate_unique_data(n, d as usize, 777);
    let ids: Vec<u64> = (1000..1000 + n as u64).collect();
    let ids_idx: Vec<faiss_next::Idx> = ids.iter().map(|&id| faiss_next::Idx::new(id)).collect();

    index.add_with_ids(&data, &ids_idx).unwrap();
    assert_eq!(index.ntotal(), n as u64);

    let offset = d as usize * 50;
    let query: Vec<f32> = data[offset..offset + d as usize].to_vec();
    let result = index.search(&query, 5).unwrap();

    assert_eq!(result.labels[0].get(), Some(1050));
}

#[test]
fn test_io_write_read_index() {
    let d = 32u32;
    let n = 100usize;
    let path = "/tmp/test_faiss_index.bin";

    cleanup_test_file(path);

    let mut index = IndexFlat::new_l2(d).unwrap();
    let data = generate_unique_data(n, d as usize, 888);

    index.add(&data).unwrap();

    write_index(&index, path).unwrap();
    assert!(Path::new(path).exists());

    let mut loaded_index = read_index(path).unwrap();
    assert_eq!(loaded_index.ntotal(), n as u64);
    assert_eq!(loaded_index.d(), d);

    let query: Vec<f32> = data[d as usize * 42..d as usize * 43].to_vec();
    let result = loaded_index.search(&query, 5).unwrap();

    assert_eq!(result.labels[0].get(), Some(42));

    cleanup_test_file(path);
}

#[test]
fn test_clustering() {
    let d = 16u32;
    let k = 5u32;

    let clustering = Clustering::new(d, k).unwrap();
    assert_eq!(clustering.k(), k as usize);
    assert_eq!(clustering.d(), d as usize);
}

#[test]
fn test_clustering_with_params() {
    let d = 8u32;
    let k = 3u32;

    let params = ClusteringParameters::default();
    let clustering = Clustering::new_with_params(d, k, &params).unwrap();
    assert_eq!(clustering.k(), k as usize);
}

#[test]
fn test_clustering_train() {
    let d = 16u32;
    let k = 5u32;
    let n = 500usize;

    let mut clustering = Clustering::new(d, k).unwrap();
    let mut index = IndexFlat::new_l2(d).unwrap();

    let data = generate_unique_data(n, d as usize, 999);

    clustering.train(n as u64, &data, &mut index).unwrap();

    let centroids = clustering.centroids();
    assert!(!centroids.is_empty());
}

#[test]
fn test_pairwise_l2_sqr() {
    use faiss_next::pairwise_l2_sqr;

    let d = 4usize;
    let x = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    let y = vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];

    let dis = pairwise_l2_sqr(d, &x, &y);

    assert_eq!(dis.len(), 4);
    assert!((dis[0] - 2.0).abs() < 1e-5);
    assert!((dis[3] - 2.0).abs() < 1e-5);
}

#[test]
fn test_inner_products() {
    use faiss_next::inner_products;

    let d = 3usize;
    let x = vec![1.0, 0.0, 0.0];
    let y = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

    let ip = inner_products(d, &x, &y);

    assert_eq!(ip.len(), 3);
    assert!((ip[0] - 1.0).abs() < 1e-5);
    assert!((ip[1] - 0.0).abs() < 1e-5);
    assert!((ip[2] - 0.0).abs() < 1e-5);
}

#[test]
fn test_norm_and_renorm() {
    use faiss_next::{norm_l2_sqr, norms_l2, renorm_l2};

    let d = 3usize;
    let mut x = vec![3.0, 4.0, 0.0, 6.0, 8.0, 0.0];

    let norm = norm_l2_sqr(&x[..d], d);
    assert!((norm - 25.0).abs() < 1e-5);

    let norms = norms_l2(&x, d);
    assert_eq!(norms.len(), 2);
    assert!((norms[0] - 5.0).abs() < 1e-5);
    assert!((norms[1] - 10.0).abs() < 1e-5);

    renorm_l2(&mut x, d);
    assert!((x[0] - 0.6).abs() < 1e-5);
    assert!((x[1] - 0.8).abs() < 1e-5);
}

#[test]
fn test_compute_residual() {
    let d = 8u32;
    let n = 100usize;

    let mut index = IndexFlat::new_l2(d).unwrap();
    let data = generate_unique_data(n, d as usize, 123);
    index.add(&data).unwrap();

    let query: Vec<f32> = data[0..d as usize].to_vec();
    let result = index.search(&query, 1).unwrap();

    let residual = index.compute_residual(&query, result.labels[0]).unwrap();
    assert_eq!(residual.len(), d as usize);
}

#[test]
fn test_search_with_params() {
    let d = 32u32;
    let n = 100usize;

    let mut index = IndexFlat::new_l2(d).unwrap();
    let data = generate_unique_data(n, d as usize, 555);
    index.add(&data).unwrap();

    let query: Vec<f32> = data[0..d as usize].to_vec();
    let params = SearchParameters::new().unwrap();
    let result = index.search_with_params(&query, 5, &params).unwrap();

    assert_eq!(result.labels.len(), 5);
    assert_eq!(result.labels[0].get(), Some(0));
}

#[test]
fn test_search_with_params_ivf() {
    let d = 32u32;
    let nlist = 10usize;
    let n = 1000usize;

    let mut index = index_factory(d, &format!("IVF{},Flat", nlist), MetricType::L2).unwrap();
    let data = generate_unique_data(n, d as usize, 666);

    index.train(&data).unwrap();
    index.add(&data).unwrap();

    let query: Vec<f32> = data[0..d as usize].to_vec();

    let mut params = SearchParametersIvf::new().unwrap();
    params.set_nprobe(5);

    let result = index.search_with_params(&query, 10, &params).unwrap();
    assert_eq!(result.labels.len(), 10);
}
