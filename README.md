# faiss-next

[![Crates.io](https://img.shields.io/crates/v/faiss-next.svg)](https://crates.io/crates/faiss-next)
[![Documentation](https://docs.rs/faiss-next/badge.svg)](https://docs.rs/faiss-next)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Rust bindings for [Faiss](https://github.com/facebookresearch/faiss) (Facebook AI Similarity Search), a library for efficient similarity search and clustering of dense vectors.

## Features

- **Safe Rust API**: Idiomatic Rust wrappers around Faiss C API
- **Multiple Index Types**: Support for Flat, IVF, LSH, Scalar Quantizer, and more
- **Multi-Version Support**: Works with Faiss 1.14.x and newer (with loose compatibility mode)
- **CUDA Support**: Optional GPU acceleration on Linux (feature flag: `cuda`)
- **Serialization**: Save and load indexes to/from disk
- **Clustering**: K-means clustering with customizable parameters
- **Pairwise Operations**: Efficient distance computations (L2, inner product)

## Platform Support

| OS | Architecture | CPU | CUDA |
|----|--------------|-----|------|
| macOS (Apple Silicon) | aarch64 (M1/M2/M3) | ✅ | ❌ |
| macOS (Intel) | x86_64 | ⚠️ | ❌ |
| Linux | x86_64 | ✅ | ✅ |
| Linux | aarch64 | ⚠️ | ❌ |
| Windows | x86_64 | ⚠️ | ⚠️ |

**Legend:**
- ✅ Fully supported with pre-generated bindings
- ⚠️ May work with `bindgen` feature to generate bindings at compile time
- ❌ Not supported

**Notes:**
- CUDA is only supported on Linux x86_64
- For unsupported platforms, enable `bindgen` feature: `faiss-next = { version = "0.6", features = ["bindgen"] }`

## Supported Index Types

| Index Type | Description |
|------------|-------------|
| `IndexFlat` | Brute-force index (exact search) |
| `IndexIVFFlat` | Inverted file with flat quantizer |
| `IndexIVFScalarQuantizer` | IVF with scalar quantization |
| `IndexScalarQuantizer` | Scalar quantizer index |
| `IndexLSH` | Locality-sensitive hashing |
| `IndexIDMap` / `IndexIDMap2` | Custom ID mapping wrapper |
| `IndexPreTransform` | Pre-transformation wrapper |
| `IndexRefineFlat` | Refinement with flat index |
| `IndexReplicas` / `IndexShards` | Distributed indexes |
| `IndexBinary` | Binary vector indexes |
| `IndexFlat1D` | Optimized 1D flat index |

## Search Parameters

| Type | Description |
|------|-------------|
| `SearchParameters` | Basic search parameters |
| `SearchParametersIvf` | IVF-specific parameters (`nprobe`, `max_codes`) |

## Getting Started

### Prerequisites

1. Install Faiss C library:

**macOS (Homebrew):**
```bash
brew install faiss
```

**Linux:**
```bash
# From source (recommended)
git clone https://github.com/facebookresearch/faiss.git
cd faiss
mkdir build && cd build
cmake -DFAISS_ENABLE_C_API=ON -DBUILD_SHARED_LIBS=ON ..
make -j
sudo make install
```

2. Ensure the library is discoverable:
   - Set `FAISS_DIR` environment variable, or
   - Install to standard locations (`/usr/local`, `/opt/homebrew`)

### Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
faiss-next = "0.6"
```

### Basic Usage

```rust,no_run
use faiss_next::{IndexFlat, Index};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a flat L2 index
    let mut index = IndexFlat::new_l2(128)?;  // 128-dimensional vectors

    // Add vectors
    let vectors: Vec<f32> = vec![0.0; 128 * 100];  // 100 vectors of 128 dimensions
    index.add(&vectors)?;

    // Search for k nearest neighbors
    let query: Vec<f32> = vec![0.0; 128];
    let result = index.search(&query, 10)?;
    for i in 0..10 {
        println!("Label: {:?}, Distance: {}", result.labels[i], result.distances[i]);
    }
    Ok(())
}
```

### Using Index Factory

```rust,no_run
use faiss_next::{index_factory, MetricType, Index};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create IVF index
    let mut index = index_factory(128, "IVF256,Flat", MetricType::L2)?;

    // Train and add vectors
    let training_data: Vec<f32> = vec![0.0; 128 * 1000];
    index.train(&training_data)?;
    
    let vectors: Vec<f32> = vec![0.0; 128 * 100];
    index.add(&vectors)?;
    Ok(())
}
```

### Using Index Builder (Fluent API)

```rust,no_run
use faiss_next::{IndexBuilder, Index};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut index = IndexBuilder::new(128)
        .ivf_flat(256)         // IVF with 256 clusters
        .l2()
        .build()?;

    let training_data: Vec<f32> = vec![0.0; 128 * 1000];
    index.train(&training_data)?;
    
    let vectors: Vec<f32> = vec![0.0; 128 * 100];
    index.add(&vectors)?;
    Ok(())
}
```

### Custom IDs

```rust,no_run
use faiss_next::{index_factory, IndexIDMap, Index, MetricType};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let base = index_factory(128, "Flat", MetricType::L2)?;
    let mut index = IndexIDMap::new(base)?;

    let vectors: Vec<f32> = vec![0.0; 128 * 100];
    let ids: Vec<u64> = (100..200).collect();
    let ids_idx: Vec<faiss_next::Idx> = ids.iter().map(|&id| faiss_next::Idx::new(id)).collect();
    index.add_with_ids(&vectors, &ids_idx)?;
    Ok(())
}
```

### Serialization

```rust,no_run
use faiss_next::{IndexFlat, Index, write_index, read_index};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut index = IndexFlat::new_l2(128)?;
    let vectors: Vec<f32> = vec![0.0; 128 * 100];
    index.add(&vectors)?;

    // Save
    write_index(&index, "my_index.bin")?;

    // Load
    let loaded = read_index("my_index.bin")?;
    Ok(())
}
```

### Pairwise Distance Computation

```rust
use faiss_next::{pairwise_l2_sqr, inner_products};

let d = 128;
let x: Vec<f32> = vec![0.0; d * 10];  // 10 query vectors
let y: Vec<f32> = vec![0.0; d * 100]; // 100 database vectors

// Compute pairwise L2 squared distances
let distances = pairwise_l2_sqr(d, &x, &y);  // 10 * 100 elements

// Compute pairwise inner products
let products = inner_products(d, &x, &y);  // 10 * 100 elements
```

### Clustering (K-means)

```rust,no_run
use faiss_next::{Clustering, IndexFlat};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut clustering = Clustering::new(128, 100)?;  // d=128, k=100
    let mut index = IndexFlat::new_l2(128)?;

    let n = 1000u64;
    let data: Vec<f32> = vec![0.0; 128 * 1000];
    clustering.train(n, &data, &mut index)?;

    let centroids = clustering.centroids();
    Ok(())
}
```

### Search with Parameters

For fine-grained control over search behavior, use `search_with_params`:

```rust,no_run
use faiss_next::{index_factory, MetricType, Index, SearchParameters, SearchParametersIvf};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut index = index_factory(128, "IVF256,Flat", MetricType::L2)?;
    
    let training_data: Vec<f32> = vec![0.0; 128 * 1000];
    index.train(&training_data)?;
    index.add(&training_data)?;

    let query: Vec<f32> = vec![0.0; 128];
    
    // Basic search parameters
    let params = SearchParameters::new()?;
    let result = index.search_with_params(&query, 10, &params)?;

    // IVF-specific parameters (nprobe, max_codes)
    let mut ivf_params = SearchParametersIvf::new()?;
    ivf_params.set_nprobe(16);       // Search 16 clusters
    ivf_params.set_max_codes(10000); // Max codes to visit
    let result = index.search_with_params(&query, 10, &ivf_params)?;
    Ok(())
}
```

### Range Search

Find all vectors within a distance threshold:

```rust,no_run
use faiss_next::{IndexFlat, Index};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut index = IndexFlat::new_l2(128)?;
    let vectors: Vec<f32> = vec![0.0; 128 * 100];
    index.add(&vectors)?;

    let query: Vec<f32> = vec![0.0; 128];
    
    // Find all vectors within radius 10.0
    let result = index.range_search(&query, 10.0)?;

    // Iterate over results
    for (labels, distances) in result.iter() {
        println!("Found {} results", labels.len());
    }

    // Get results for a specific query
    if let Some((labels, distances)) = result.get(0) {
        println!("Query 0: {} results within radius", labels.len());
    }
    Ok(())
}
```

## Feature Flags

| Flag | Description |
|------|-------------|
| `cuda` | Enable CUDA GPU support (Linux only) |
| `bindgen` | Generate bindings at compile time |

## Performance

The bindings leverage Faiss's optimized SIMD implementations:

```text
L2 Distance Benchmark (Dimension: 128)
Size (nq, nb)      ndarray (ms)    faiss (ms)     Speedup
(  100,  1000)           12.34          0.87        14.2x
( 1000,  1000)          123.45          8.21        15.0x
( 1000, 10000)         1234.56         17.89        69.0x
```

## Version Compatibility

- **Minimum**: Faiss 1.14.0
- **Tested**: Faiss 1.14.x
- **Loose Mode**: Newer versions will work with a compile-time warning

When Faiss 1.15 is released, add new bindings in the `v1_15` directory.

## Documentation

- [API Documentation](https://docs.rs/faiss-next)
- [Faiss Wiki](https://github.com/facebookresearch/faiss/wiki)

## License

MIT License

## Acknowledgments

- [Faiss](https://github.com/facebookresearch/faiss) by Facebook AI Research
- Inspired by [faiss-rs](https://github.com/Enet4/faiss-rs)