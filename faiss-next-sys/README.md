# faiss-next-sys

`faiss-next-sys` wrap `c_api` of `faiss c_api` into `rust` with `bindgen`.

`faiss-next-sys` also build a static version of `faiss`.

Currently supported `faiss` version is `v1.7.4`

## Setup build environment

### `Macos` setup

- `llvm` and `cmake` is required to build faiss on mac, install use `brew`, `clagn/clang++` provided by `xcode` does not support `openmp`, so we need `llvm`
	```shell
	brew install llvm cmake
	```

### `Linux` setup 

- OS: `Ubunt 22.04 LTS` used.
- `cuda-11`  `cmake` should be installed.
- `intel-mkl` `intel-mpi` should be install according [intel](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html?operatingsystem=linux&distributions=aptpackagemanager) documents. Do not install `mkl` provided by `ubuntu`. static link faiss will not success.


### `Windows` setup 

TBD