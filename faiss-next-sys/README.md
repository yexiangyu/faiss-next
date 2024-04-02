# `faiss-next-sys`

`faiss-next-sys` is generated bindings of `faiss.h`.

## Supported platform

| OS      | x86_64 | Apple Silicon | CUDA |
|---------|--------|---------------|------|
| Linux   |  Yes   | N/A           |  Yes |
| Macos   |  N/A   | Yes           |  N/A |
| Windows |  Yes   | N/A           |  Yes |

## Environment variable for linking

- `FAISS_INCLUDE_DIR`: include dir of `faiss`, on `Windows`, the default folder is `c:\tools\faiss\include`
- `FAISS_LIB_DIR`: library dir of `faiss`, on `Windows`, the default folder is `c:\tools\faiss\lib`
- `CUDA_PATH`:  `cuda` installation dir, if `gpu` feature is enabled, `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3`.

## Update bindings of `faiss-next-sys` 

```shell
cd faiss-next-sys
# generate  bindinge for cpu
cargo build --features bindgen
# generate  bindinge for gpu
cargo build --features bindgen, gpu
```

## About `faiss` installation

 - `faiss` need to be built with `c_api` feature enabled, following `cmake` command will build `c_api` enabled on `Windows`.

 ```shell
 cmake -B build -Ax64 -DFAISS_ENABLE_C_API=ON -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release -DFAISS_ENABLE_GPU=ON -DFAISS_ENABLE_PYTHON=OFF -DBUILD_TESTING=OFF

# build 
cmake --build build --config Release

# install
cmake --install build --prefix c:\tools\faiss
 ```

- `static` linking of `faiss` lib is not supported.
- `homebrew` installed `faiss` do not compiled with `c_api` supported.