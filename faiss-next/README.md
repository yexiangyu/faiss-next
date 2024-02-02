# faiss-next

`faiss-next` is a simple rust bindings for [facebookresearch/faiss](https://github.com/facebookresearch/faiss). This crate is is inspired by [Enet4/faiss-rs](https://github.com/Enet4/faiss-rs).

`Windows`, `Linux` and `Macos` is supported. `facebookresearch/faiss` `v1.7.4` is wrapped currently.

`faiss-next` requires `faiss` compiled with `FAISS_ENABLE_C_API=ON` and `BUILD_SHARED_LIBS=ON` in advance. Please checkout [`README.md`](https://github.com/yexiangyu/faiss-next/blob/main/faiss-next-sys/README.md) of `faiss-next-sys` for further info about building `faiss` from source.

## Installation

Before linking with `faiss-next`, env variable `FAISS_DIR` should set and point to the dir `faiss` installed.  If `FAISS_DIR` is not set,  `build.rs` will search `/usr` or `/usr/local` or `$HOME/faiss` (`%USERPROFILE%/faiss` on `windows`) for library and include heeders by default. 


```toml
[dependencies]
faiss-next = {version = "*", features = ["gpu"] }
```

## Tutorial

```rust
use faiss_next::*;

use ndarray::{s, Array2};
use ndarray_rand::*;


fn main() {

	//create index
	let mut index = index_factory(128, "Flat", FaissMetricType::METRIC_L2).expect("failed to create cpu index");

	//create some random feature
	let feats = Array2::random((1024, 128), rand::distributions::Uniform::new(0., 1.));

	//get query from position 42
	let query = feats.slice(s![42..43, ..]);

	//add features in index
	index.add(feats.as_slice_memory_order().unwrap()).expect("failed to add feature");

	//do the search
	let ret = index.search(query.as_slice_memory_order().unwrap(), 1).expect("failed to search");
	assert_eq!(ret.labels[0], 42i64);

	//move index from cpu to gpu, only available when gpu feature is enabled
	#[cfg(feature = "gpu")]
	{
		let index = index.into_gpu(0).expect("failed to move index to gpu");
		let ret = index.search(query.as_slice_memory_order().unwrap(), 1).expect("failed to search");
		assert_eq!(ret.labels[0], 42i64);
	}
}
```