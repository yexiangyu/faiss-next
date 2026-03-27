use faiss_next::{GpuIndexImpl, GpuResources, Index, IndexFlat};

fn main() {
    println!("=== Faiss GPU Test ===\n");

    let d = 128u32;
    let n = 1000usize;

    println!("Creating CPU index with d={}...", d);
    let mut index = IndexFlat::new_l2(d).expect("Failed to create CPU index");

    println!("Generating {} random vectors...", n);
    let data: Vec<f32> = (0..d as usize * n)
        .map(|i| (i % 1000) as f32 / 1000.0)
        .collect();

    println!("Adding vectors to CPU index...");
    index.add(&data).expect("Failed to add vectors");
    println!("CPU index has {} vectors\n", index.ntotal());

    println!("Creating GPU resources...");
    match GpuResources::new() {
        Ok(resources) => {
            println!("GPU resources created successfully!\n");

            println!("Moving index to GPU (device 0)...");
            match GpuIndexImpl::from_cpu(&index, resources, 0) {
                Ok(mut gpu_index) => {
                    println!("GPU index created successfully!");
                    println!("GPU index has {} vectors\n", gpu_index.ntotal());

                    println!("Performing search on GPU index...");
                    let query: Vec<f32> =
                        (0..d as usize).map(|i| (i % 100) as f32 / 100.0).collect();
                    match gpu_index.search(&query, 5) {
                        Ok(result) => {
                            println!("Search results (top 5):");
                            for (i, (label, distance)) in result
                                .labels
                                .iter()
                                .zip(result.distances.iter())
                                .enumerate()
                            {
                                if let Some(l) = label.get() {
                                    println!("  {}: label={}, distance={:.4}", i + 1, l, distance);
                                }
                            }
                        }
                        Err(e) => println!("Search failed: {:?}", e),
                    }

                    println!("\nMoving index back to CPU...");
                    match gpu_index.to_cpu() {
                        Ok(cpu_index) => {
                            println!("Index moved back to CPU successfully!");
                            println!("CPU index has {} vectors", cpu_index.ntotal());
                        }
                        Err(e) => println!("Failed to move index back to CPU: {:?}", e),
                    }
                }
                Err(e) => {
                    println!("Failed to create GPU index: {:?}", e);
                    println!("\nNote: This error may occur if:");
                    println!("  - No CUDA-capable GPU is available");
                    println!("  - CUDA driver is not installed");
                    println!("  - Faiss was built without CUDA support");
                }
            }
        }
        Err(e) => {
            println!("Failed to create GPU resources: {:?}", e);
        }
    }

    println!("\n=== Test Complete ===");
}
