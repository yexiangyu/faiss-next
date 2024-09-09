# `faiss-next`

- `faiss-next` is a rust binding for `facebookresearch/faiss` c_api use `bindgen`
- support `facebookresearch/faiss` version `v1.8.0`, other version ***NOT*** verified.
- only shared library: `libfaiss.[so|dylib]`, `libfaiss_c.[so|dylib]` or `faiss.dll`, `faiss_c.dll` is supported. Static linking with `*.a` is a painful experience with 3rd library dependency with `mkl` or `cuda`.
- current status of implemenetation on different platform: 

| OS      | Arch          | Available |
| ------- | ------------- | --------- |
| MacOS   | aarch64       | yes       |
| MacOS   | x66_64        | no        |
| Linux   | x86_64        | yes       |
| Linux   | x86_64 + cuda | yes       |
| Windows | x86_64        | no        |
| Windows | x86_64 + cuda | no        |

- Extend `function` not available in `c_api`
    - `faiss_index_binary_factory`

## pre-requirement

`facebookresearch/faiss` installation needed.

1. Clone `facebookreseach/faiss` source code to local, then checkout the tag `v1.8.0`:
    ```shell
    git clone https://github.com/facebookresearch/faiss
    cd faiss
    git checkout -b v1.8.0 v1.8.0
    ```
2. Build `facebookreseach/faiss`
    - On `MacOS`:
      - `facebookreseach/faiss` could be installed by `brew install faiss`, but `c_api` shared libaries is missing. to build `libfaiss_c.dylib`, `cd c_api` in the source tree of `facebookreseach/faiss` and modify `CMakeLists.txt` like following:
      ```cmake
      ...
      set(CMAKE_C_STANDARD 11)
      # append after 
      set(CMAKE_CXX_STANDARD 11)
      include_directories("/opt/homebrew/opt/faiss/include")
      link_directories("/opt/homebrew/opt/faiss/lib")
      ...
      # change build target from *.a to *.dylib
      add_library(faiss_c SHARED ${FAISS_C_SRC})
      ```
      - build `libfaiss.dylib` by run `cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build` 
      - copy `libfaiss_c.dylib` to `/opt/homebrew/opt/faiss/lib`
      - linking path search will done automatically by `build.rs` in `faiss-next-sys` crate on `MacOS`.
    - On `Linux`, just follow [official guide](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md) to build shared library or:
        ```shell

        # cd faiss source tree
        cd faiss

        # configure
        cmake -B build -DFAISS_ENABLE_C_API=ON -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release -DFAISS_ENABLE_GPU=ON -DFAISS_ENABLE_PYTHON=OFF -DBUILD_TESTING=OFF

        # build
        cd build && make -j && cd .. && sudo cmake --install . && sudo cp c_api/libfaiss_c.so /usr/local/lib
        ```
    - On `Windows` (TODO)

## Generate bindings
- By enable feature `bindgen` in `faiss-next-sys` will generate bindings.
    ```shell
    cd faiss-next-sys
    cargo build --features bindgen
    ```
    this will create a bindings like `macos/aarch64.rs`

## Getting started

```rust
use faiss_next::prelude::*;
use ndarray_rand::{rand_distr::Uniform, RandomExt};
use ndarray::Array2;
use itertools::Itertools;

// define base size and dimension
let d = 128;
let n = 1024;

// create base need to be search
let base = Array2::<f32>::random((n, d), Uniform::new(-1.0, 1.0));

// use vector #42 as answer;
let query = base.slice(ndarray::s![42..43, ..]);

// create index
let mut index = faiss_index_factory(d as i32, "Flat,IDMap", FaissMetricType::METRIC_L2).unwrap(); 

// add vector with id start from 100
index.add_with_ids(
    base.as_slice().unwrap(),
    (100..100 + n as i64).collect_vec(),
).unwrap();

let mut distances = vec![0.0];
let mut labels = vec![-1];

index.search(query.as_slice().unwrap(), 1, &mut distances, &mut labels).unwrap();

assert_eq!(labels, &[142])

```

## Running with `CUDA`