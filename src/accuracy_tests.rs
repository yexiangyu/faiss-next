#[cfg(test)]
mod tests {
    use crate::bindings::FaissMetricType;
    use crate::index::Index;
    use crate::index_factory::index_factory;
    use crate::index_flat::IndexFlat;
    use crate::index_ivf::IndexIVF;
    use crate::index_lsh::IndexLSH;
    use crate::traits::FaissIVFIndex;
    use ndarray::Array2;
    use ndarray_rand::{rand_distr::Uniform, RandomExt};

    fn generate_random_vectors(n: usize, d: usize) -> Array2<f32> {
        Array2::random((n, d), Uniform::new(-1.0f32, 1.0f32))
    }

    fn generate_random_binary_vectors(n: usize, d_bytes: usize) -> Array2<u8> {
        Array2::random((n, d_bytes), Uniform::new(0u8, 255u8))
    }

    fn compute_l2_distance_sq(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f32>()
    }

    fn compute_inner_product(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    fn compute_hamming_distance(a: &[u8], b: &[u8]) -> i32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x ^ y).count_ones() as i32)
            .sum()
    }

    fn brute_force_search_l2(
        data: &[f32],
        query: &[f32],
        n: usize,
        d: usize,
        k: usize,
    ) -> Vec<(i64, f32)> {
        let mut results: Vec<(i64, f32)> = (0..n)
            .map(|i| {
                let vector = &data[i * d..(i + 1) * d];
                let dist = compute_l2_distance_sq(query, vector);
                (i as i64, dist)
            })
            .collect();
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        results.into_iter().take(k).collect()
    }

    fn brute_force_search_inner_product(
        data: &[f32],
        query: &[f32],
        n: usize,
        d: usize,
        k: usize,
    ) -> Vec<(i64, f32)> {
        let mut results: Vec<(i64, f32)> = (0..n)
            .map(|i| {
                let vector = &data[i * d..(i + 1) * d];
                let dist = -compute_inner_product(query, vector);
                (i as i64, dist)
            })
            .collect();
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        results.into_iter().take(k).collect()
    }

    fn brute_force_search_hamming(
        data: &[u8],
        query: &[u8],
        n: usize,
        d_bytes: usize,
        k: usize,
    ) -> Vec<(i64, i32)> {
        let mut results: Vec<(i64, i32)> = (0..n)
            .map(|i| {
                let vector = &data[i * d_bytes..(i + 1) * d_bytes];
                let dist = compute_hamming_distance(query, vector);
                (i as i64, dist)
            })
            .collect();
        results.sort_by(|a, b| a.1.cmp(&b.1));
        results.into_iter().take(k).collect()
    }

    #[test]
    fn test_flat_index_topk_correctness_l2() {
        let d = 32;
        let n = 500;
        let k = 10;

        let mut index = IndexFlat::new(d as i32, FaissMetricType::METRIC_L2).unwrap();

        let data = generate_random_vectors(n, d);
        let data_slice: Vec<f32> = data.iter().copied().collect();

        index.add(n as i64, &data_slice).unwrap();

        let n_queries = 5;
        for q_idx in 0..n_queries {
            let query: Vec<f32> = data.row(q_idx).to_owned().iter().cloned().collect();

            let mut distances = vec![0.0f32; k];
            let mut labels = vec![-1i64; k];
            index
                .search(1, &query, k as i64, &mut distances, &mut labels)
                .unwrap();

            let expected = brute_force_search_l2(&data_slice, &query, n, d, k);

            for i in 0..k {
                assert_eq!(
                    labels[i], expected[i].0,
                    "Query {}: Wrong label at position {}",
                    q_idx, i
                );
                assert!(
                    (distances[i] - expected[i].1).abs() < 1e-4,
                    "Query {}: Wrong distance at position {}. Expected {}, got {}",
                    q_idx,
                    i,
                    expected[i].1,
                    distances[i]
                );
            }
        }
    }

    #[test]
    fn test_flat_index_topk_correctness_inner_product() {
        let d = 32;
        let n = 500;
        let k = 10;

        let mut index = IndexFlat::new(d as i32, FaissMetricType::METRIC_INNER_PRODUCT).unwrap();

        let data = generate_random_vectors(n, d);
        let data_slice: Vec<f32> = data.iter().copied().collect();

        index.add(n as i64, &data_slice).unwrap();

        let n_queries = 5;
        for q_idx in 0..n_queries {
            let query: Vec<f32> = data.row(q_idx).to_owned().iter().cloned().collect();

            let mut distances = vec![0.0f32; k];
            let mut labels = vec![-1i64; k];
            index
                .search(1, &query, k as i64, &mut distances, &mut labels)
                .unwrap();

            let expected = brute_force_search_inner_product(&data_slice, &query, n, d, k);

            for i in 0..k {
                assert_eq!(
                    labels[i], expected[i].0,
                    "Query {}: Wrong label at position {}",
                    q_idx, i
                );
            }
        }
    }

    #[test]
    fn test_distance_calculation_accuracy_l2() {
        let d = 16;
        let n = 10;

        let mut index = IndexFlat::new(d as i32, FaissMetricType::METRIC_L2).unwrap();

        let data = generate_random_vectors(n, d);
        let data_slice: Vec<f32> = data.iter().copied().collect();

        index.add(n as i64, &data_slice).unwrap();

        for test_idx in 0..3 {
            let query: Vec<f32> = data.row(test_idx).to_owned().iter().cloned().collect();
            let mut distances = vec![0.0f32; n];
            let mut labels = vec![-1i64; n];

            index
                .search(1, &query, n as i64, &mut distances, &mut labels)
                .unwrap();

            let mut id_to_dist: std::collections::HashMap<i64, f32> =
                std::collections::HashMap::new();
            for i in 0..n {
                id_to_dist.insert(labels[i], distances[i]);
            }

            for i in 0..n {
                let row_i: Vec<f32> = data.row(i).to_owned().iter().cloned().collect();
                let expected_dist = compute_l2_distance_sq(&query, &row_i);
                let returned_dist = id_to_dist[&(i as i64)];
                assert!(
                    (returned_dist - expected_dist).abs() < 1e-4,
                    "Query {}: Distance mismatch for vector {}: expected {}, got {}",
                    test_idx,
                    i,
                    expected_dist,
                    returned_dist
                );
            }
        }
    }

    #[test]
    fn test_distance_calculation_accuracy_inner_product() {
        let d = 16;
        let n = 10;

        let mut index = IndexFlat::new(d as i32, FaissMetricType::METRIC_INNER_PRODUCT).unwrap();

        let data = generate_random_vectors(n, d);
        let data_slice: Vec<f32> = data.iter().copied().collect();

        index.add(n as i64, &data_slice).unwrap();

        for test_idx in 0..3 {
            let query: Vec<f32> = data.row(test_idx).to_owned().iter().cloned().collect();
            let mut distances = vec![0.0f32; n];
            let mut labels = vec![-1i64; n];

            index
                .search(1, &query, n as i64, &mut distances, &mut labels)
                .unwrap();

            let mut id_to_dist: std::collections::HashMap<i64, f32> =
                std::collections::HashMap::new();
            for i in 0..n {
                id_to_dist.insert(labels[i], distances[i]);
            }

            for i in 0..n {
                let row_i: Vec<f32> = data.row(i).to_owned().iter().cloned().collect();
                let expected_ip = compute_inner_product(&query, &row_i);
                let returned_dist = id_to_dist[&(i as i64)];
                assert!(
                    (returned_dist - expected_ip).abs() < 1e-4,
                    "Query {}: Inner product mismatch for vector {}: expected {}, got {}",
                    test_idx,
                    i,
                    expected_ip,
                    returned_dist
                );
            }
        }
    }

    #[test]
    fn test_binary_index_topk_correctness() {
        let d = 64;
        let d_bytes = d / 8;
        let n = 200;
        let k = 10;

        let mut index = crate::index_binary::IndexBinary::new_flat(d as i32).unwrap();

        let data = generate_random_binary_vectors(n, d_bytes);
        let data_slice: Vec<u8> = data.iter().copied().collect();

        index.add(n as i64, &data_slice).unwrap();

        let n_queries = 5;
        for q_idx in 0..n_queries {
            let query: Vec<u8> = data.row(q_idx).to_owned().iter().cloned().collect();

            let mut distances = vec![0i32; k];
            let mut labels = vec![-1i64; k];
            index
                .search(1, &query, k as i64, &mut distances, &mut labels)
                .unwrap();

            let expected = brute_force_search_hamming(&data_slice, &query, n, d_bytes, k);

            for i in 0..k {
                assert_eq!(
                    labels[i], expected[i].0,
                    "Query {}: Wrong label at position {}",
                    q_idx, i
                );
                assert_eq!(
                    distances[i], expected[i].1,
                    "Query {}: Wrong Hamming distance at position {}",
                    q_idx, i
                );
            }
        }
    }

    fn compute_recall_at_k(
        index: &mut Index,
        ground_truth: &[Vec<i64>],
        queries: &[f32],
        k: usize,
        n_queries: usize,
        d: usize,
    ) -> f32 {
        let mut total_recall = 0.0;

        for q_idx in 0..n_queries {
            let query = &queries[q_idx * d..(q_idx + 1) * d];
            let mut distances = vec![0.0f32; k];
            let mut labels = vec![-1i64; k];

            index
                .search(1, query, k as i64, &mut distances, &mut labels)
                .unwrap();

            let retrieved: std::collections::HashSet<i64> = labels.iter().cloned().collect();
            let relevant: std::collections::HashSet<i64> =
                ground_truth[q_idx].iter().cloned().collect();

            let hits = retrieved.intersection(&relevant).count() as f32;
            total_recall += hits / k as f32;
        }

        total_recall / n_queries as f32
    }

    #[test]
    fn test_recall_ivf_index() {
        let d = 32;
        let n = 1000;
        let n_queries = 20;
        let k = 10;
        let nlist = 20;

        let data = generate_random_vectors(n, d);
        let data_slice: Vec<f32> = data.iter().copied().collect();

        let queries = data.slice(ndarray::s![0..n_queries, ..]).to_owned();
        let queries_slice: Vec<f32> = queries.iter().copied().collect();

        let mut ground_truth: Vec<Vec<i64>> = Vec::with_capacity(n_queries);
        for q_idx in 0..n_queries {
            let query: Vec<f32> = queries.row(q_idx).to_owned().iter().cloned().collect();
            let expected = brute_force_search_l2(&data_slice, &query, n, d, k);
            ground_truth.push(expected.iter().map(|(id, _)| *id).collect());
        }

        let index = index_factory(
            d as i32,
            &format!("IVF{},Flat", nlist),
            FaissMetricType::METRIC_L2,
        )
        .unwrap();
        let mut ivf_index = IndexIVF::from_index(index);

        ivf_index.train(n as i64, &data_slice).unwrap();
        ivf_index.add(n as i64, &data_slice).unwrap();

        ivf_index.set_nprobe(nlist);

        let mut total_recall = 0.0;
        for q_idx in 0..n_queries {
            let query = &queries_slice[q_idx * d..(q_idx + 1) * d];
            let mut distances = vec![0.0f32; k];
            let mut labels = vec![-1i64; k];

            ivf_index
                .search(1, query, k as i64, &mut distances, &mut labels)
                .unwrap();

            let retrieved: std::collections::HashSet<i64> = labels.iter().cloned().collect();
            let relevant: std::collections::HashSet<i64> =
                ground_truth[q_idx].iter().cloned().collect();

            let hits = retrieved.intersection(&relevant).count() as f32;
            total_recall += hits / k as f32;
        }
        let recall = total_recall / n_queries as f32;

        println!(
            "IVF Index (nlist={}, nprobe={}): Recall@{} = {:.4}",
            nlist, nlist, k, recall
        );
        assert!(recall > 0.9, "Recall@{} should be > 0.9, got {}", k, recall);
    }

    #[test]
    fn test_recall_lsh_index() {
        let d = 16;
        let n = 500;
        let n_queries = 10;
        let k = 10;
        let nbits = 128;

        let data = generate_random_vectors(n, d);
        let data_slice: Vec<f32> = data.iter().copied().collect();

        let queries = data.slice(ndarray::s![0..n_queries, ..]).to_owned();
        let queries_slice: Vec<f32> = queries.iter().copied().collect();

        let mut ground_truth: Vec<Vec<i64>> = Vec::with_capacity(n_queries);
        for q_idx in 0..n_queries {
            let query: Vec<f32> = queries.row(q_idx).to_owned().iter().cloned().collect();
            let expected = brute_force_search_l2(&data_slice, &query, n, d, k);
            ground_truth.push(expected.iter().map(|(id, _)| *id).collect());
        }

        let mut index = IndexLSH::new(d as i32, nbits).unwrap();
        index.add(n as i64, &data_slice).unwrap();

        let mut index_wrapper = Index { inner: index.inner };
        std::mem::forget(index);
        let recall = compute_recall_at_k(
            &mut index_wrapper,
            &ground_truth,
            &queries_slice,
            k,
            n_queries,
            d,
        );

        println!("LSH Index (nbits={}): Recall@{} = {:.4}", nbits, k, recall);
        assert!(recall > 0.3, "Recall@{} should be > 0.3, got {}", k, recall);
    }

    #[test]
    fn test_recall_pq_index() {
        let d = 32;
        let n = 1000;
        let n_queries = 20;
        let k = 10;

        let data = generate_random_vectors(n, d);
        let data_slice: Vec<f32> = data.iter().copied().collect();

        let queries = data.slice(ndarray::s![0..n_queries, ..]).to_owned();
        let queries_slice: Vec<f32> = queries.iter().copied().collect();

        let mut ground_truth: Vec<Vec<i64>> = Vec::with_capacity(n_queries);
        for q_idx in 0..n_queries {
            let query: Vec<f32> = queries.row(q_idx).to_owned().iter().cloned().collect();
            let expected = brute_force_search_l2(&data_slice, &query, n, d, k);
            ground_truth.push(expected.iter().map(|(id, _)| *id).collect());
        }

        let mut index = index_factory(d as i32, "PQ8", FaissMetricType::METRIC_L2).unwrap();

        index.train(n as i64, &data_slice).unwrap();
        index.add(n as i64, &data_slice).unwrap();

        let recall =
            compute_recall_at_k(&mut index, &ground_truth, &queries_slice, k, n_queries, d);

        println!("PQ Index: Recall@{} = {:.4}", k, recall);
        assert!(recall > 0.5, "Recall@{} should be > 0.5, got {}", k, recall);
    }

    #[test]
    fn test_recall_hnsw_index() {
        let d = 32;
        let n = 500;
        let n_queries = 20;
        let k = 10;

        let data = generate_random_vectors(n, d);
        let data_slice: Vec<f32> = data.iter().copied().collect();

        let queries = data.slice(ndarray::s![0..n_queries, ..]).to_owned();
        let queries_slice: Vec<f32> = queries.iter().copied().collect();

        let mut ground_truth: Vec<Vec<i64>> = Vec::with_capacity(n_queries);
        for q_idx in 0..n_queries {
            let query: Vec<f32> = queries.row(q_idx).to_owned().iter().cloned().collect();
            let expected = brute_force_search_l2(&data_slice, &query, n, d, k);
            ground_truth.push(expected.iter().map(|(id, _)| *id).collect());
        }

        let mut index = index_factory(d as i32, "HNSW32", FaissMetricType::METRIC_L2).unwrap();

        index.add(n as i64, &data_slice).unwrap();

        let recall =
            compute_recall_at_k(&mut index, &ground_truth, &queries_slice, k, n_queries, d);

        println!("HNSW Index: Recall@{} = {:.4}", k, recall);
        assert!(recall > 0.9, "Recall@{} should be > 0.9, got {}", k, recall);
    }

    #[test]
    fn test_multiple_queries_accuracy() {
        let d = 32;
        let n = 500;
        let n_queries = 50;
        let k = 10;

        let mut index = IndexFlat::new(d as i32, FaissMetricType::METRIC_L2).unwrap();

        let data = generate_random_vectors(n, d);
        let data_slice: Vec<f32> = data.iter().copied().collect();

        index.add(n as i64, &data_slice).unwrap();

        let queries = data.slice(ndarray::s![0..n_queries, ..]).to_owned();
        let queries_slice: Vec<f32> = queries.iter().copied().collect();

        let mut distances = vec![0.0f32; n_queries * k];
        let mut labels = vec![-1i64; n_queries * k];

        index
            .search(
                n_queries as i64,
                &queries_slice,
                k as i64,
                &mut distances,
                &mut labels,
            )
            .unwrap();

        let mut correct = 0;
        let total = n_queries * k;

        for q_idx in 0..n_queries {
            let query: Vec<f32> = queries.row(q_idx).to_owned().iter().cloned().collect();
            let expected = brute_force_search_l2(&data_slice, &query, n, d, k);

            for i in 0..k {
                let returned_label = labels[q_idx * k + i];
                let expected_label = expected[i].0;
                if returned_label == expected_label {
                    correct += 1;
                }
            }
        }

        let accuracy = correct as f32 / total as f32;
        println!(
            "Multi-query accuracy: {:.4} ({}/{})",
            accuracy, correct, total
        );
        assert!(
            accuracy > 0.99,
            "Multi-query accuracy should be > 0.99, got {}",
            accuracy
        );
    }

    #[test]
    fn test_exact_match_self_query() {
        let d = 32;
        let n = 100;
        let k = 1;

        let mut index = IndexFlat::new(d as i32, FaissMetricType::METRIC_L2).unwrap();

        let data = generate_random_vectors(n, d);
        let data_slice: Vec<f32> = data.iter().copied().collect();

        index.add(n as i64, &data_slice).unwrap();

        for i in 0..20 {
            let query: Vec<f32> = data.row(i).to_owned().iter().cloned().collect();
            let mut distances = vec![0.0f32; k];
            let mut labels = vec![-1i64; k];

            index
                .search(1, &query, k as i64, &mut distances, &mut labels)
                .unwrap();

            assert_eq!(
                labels[0], i as i64,
                "Self-query should return the same vector"
            );
            assert!(
                distances[0] < 1e-5,
                "Self-query distance should be ~0, got {}",
                distances[0]
            );
        }
    }
}
