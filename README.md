# faiss-next

`faiss-next` is a simple rust bindings for [facebookresearch/faiss](https://github.com/facebookresearch/faiss). This crate is is inspired by [Enet4/faiss-rs](https://github.com/Enet4/faiss-rs).

`Windows`, `Linux` and `Macos` is supported.

Currently `facebookresearch/faiss` `v1.7.4` is wrapped.

## Installation

`faiss-next` requires `faiss` compiled with `FAISS_ENABLE_C_API=ON` and `BUILD_SHARED_LIBS=ON` in advance. [facebookresearch/faiss](https://github.com/facebookresearch/faiss) provides [document](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md) about how to build it from source. A short guide is provided at the end of `README.md`

put `faiss-next` in your `Cargo.toml` file

```toml
[dependencies]
faiss-next = {version = "*", features = ["gpu"] }
```

When linking against `faiss-next`, env variable `FAISS_DIR` will point to the dir `faiss` installed.  If `FAISS_DIR` is not set,  `build.rs` will search `/usr` or `/usr/local` or `$HOME/faiss` for library and include heeders by default. 

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

## Build `faiss` from source

`faiss-next` requires `faiss` compiled with `FAISS_ENABLE_C_API=ON` and `BUILD_SHARED_LIBS=ON` in advance. [facebookresearch/faiss](https://github.com/facebookresearch/faiss) provides [document](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md) about how to build it from source, here is a quick howto about buiding `faiss` from source.

On `windows`, build will fail because `msvc` c++ compiler compatable [issue](https://github.com/facebookresearch/faiss/issues/2985), so I created a personal heck to make things work, https://github.com/yexiangyu/faiss/archive/refs/heads/v1.7.4-win.zip, if `windows` is not used, just clone `faiss` and checkout `v1.7.4` branch will just work.

Download the [zip](https://github.com/yexiangyu/faiss/archive/refs/heads/v1.7.4-win.zip), unzip then unzip it.

### `MacOS`
`xcode` and [`brew`](https://brew.sh) needed, install in advance.
 ```shell
 # install cmake openblas and llvm
 brew install cmake openblas llvm
 
 # configure
 cmake -B build -DCMAKE_C_COMPILER=/opt/homebrew/opt/llvm/bin/clang -DCMAKE_CXX_COMPILER=/opt/homebrew/opt/llvm/bin/clang++  -DFAISS_ENABLE_C_API=ON -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF -DBUILD_TESTING=OFF
 
 # compile
 cmake --build build --config Release
 
 # install
 cmake --install build --prefix=$HOME/faiss
 cp build/c_api/libfaiss_c.dylib $HOME/faiss/lib/
```

### `Linux`

`gcc`, `cmake`, [`intelmkl`](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html), `cuda` needed, install in advance.

```shell
# configure
cmake -B build -DFAISS_ENABLE_C_API=ON -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release -DFAISS_ENABLE_GPU=ON -DFAISS_ENABLE_PYTHON=OFF -DBUILD_TESTING=OFF

# compile
 cmake --build build --config Release

# install
 cmake --install build --prefix=$HOME/faiss
 cp build/c_api/libfaiss_c.so $HOME/faiss/lib/
```
### `Windows`
`Visual Studio 2022`, `cmake`, [`intelmkl`](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html), `cuda` needed, install in advance.

```shell
# configure
cmake -B build -DFAISS_ENABLE_C_API=ON -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release -DFAISS_ENABLE_GPU=ON -DFAISS_ENABLE_PYTHON=OFF -DBUILD_TESTING=OFF

# compile
 cmake --build build --config Release

# install
cmake --install build --prefix=%USERPROFILE%\faiss
copy build\c_api\Release\faiss_c.dll %USERPROFILE%\faiss\bin
copy build\c_api\Release\faiss_c.lib %USERPROFILE%\faiss\lib\
```

`static` linking is not supported.



