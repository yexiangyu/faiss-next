# ```faiss-next```

```faiss-next``` is a `c_api` wrapper for `facebookresearch/faiss`

## Integrated `faiss-next` into projects

```rust
// use faiss
use faiss_next as faiss;
use faiss::prelude::*;
```

## Tutorial

```rust
use faiss_next as faiss;
use faiss::prelude::*;
use faiss::index::SearchParameters;
use faiss::error::Result;
use ndarray::Array2;
use ndarray_rand::{RandomExt, rand_distr::Uniform};

fn index_search() -> Result<()>
{

    let d = 128;
    let n = 1024;
    let k = 1;

    // create random array as base
    let base = Array2::<f32>::random([n, d], Uniform::new(-1.0, 1.0));

    // array as &[f32]
    let base = base.as_slice_memory_order().expect("not standard memory layout");

    // use vector #42 as query
    let query = &base[42*d..43*d];

    // create flat index as d=128, with L2 metric
    let mut index = faiss::index::flat::IndexFlat::new(d, MetricType::L2)?;

    // add base to vector without id
    index.add(base, Option::<&[i64]>::None)?;

    let mut scores = vec![0.0f32; k as usize];
    let mut labels = vec![0i64; k as usize];

    if cfg!(feature = "gpu")
    {
        // use gpu by enable feature "gpu"
        use faiss::gpu::index::IndexGpuImpl;
        use faiss::gpu::standard_resources::StandardGpuResources;
        use faiss::gpu::cloner_options::GpuClonerOptions;
        let providers = vec![StandardGpuResources::new()?];
        let devices = vec![0];
        let index = IndexGpuImpl::new(providers, devices, &index, Option::<GpuClonerOptions>::None)?;
        index.search(query, k, &mut scores, &mut labels, Option::<SearchParameters>::None)?;
    }
    else
    {
        index.search(query, k, &mut scores, &mut labels, Option::<SearchParameters>::None)?;
    }

    assert_eq!(scores, [0.0]);
    assert_eq!(labels, [42]);

    Ok(())
}

index_search().expect("search failed");
```
