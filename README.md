# faiss-next

[faiss-next](https://github.com/yexiangyu/faiss-next/blob/main/faiss-next/README.md): rust wrapper for [facebookresearch/faiss](https://github.com/facebookresearch/faiss), inspired by [Enet4/faiss-rs](https://github.com/Enet4/faiss-rs)

`faiss-next` support `windows`, `linux` and `macos`. 

[facebookresearch/faiss](https://github.com/facebookresearch/faiss) `v1.7.4` is currently wrapped. 

# Installation

`faiss-next` requires `faiss` compiled with `FAISS_ENABLE_C_API=ON` and `BUILD_SHARED_LIBS=ON` in advance. [facebookresearch/faiss](https://github.com/facebookresearch/faiss) provides [document](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md) about how to build it from source, a quick howto about buiding `faiss` from source is at the end.

put `faiss-next` in your `Cargo.toml` file

```toml
[dependencies]
faiss-next = "*"
```
when linking against `faiss-next`, env variable `FAISS_DIR` need to specified, if `faiss` is not installed in standard dir like `/usr` or `/usr/local`. `build.rs` from `faiss-next-sys` will search `faiss` folder under current user's `home` directory for `faiss` installation by default.

# Features

- `bindgen`: re-generate `ffi` bindings for `faiss.h`, only `faiss` version updated in future.
- `gpu`: enable support with `faiss` is compiled with `cuda` supported.

The bindings is already created, if re-created required. please enable `feature`: `bindgen`.

# Build `faiss` from source

`faiss-next` requires `faiss` compiled with `FAISS_ENABLE_C_API=ON` and `BUILD_SHARED_LIBS=ON` in advance. [facebookresearch/faiss](https://github.com/facebookresearch/faiss) provides [document](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md) about how to build it from source, here is a quick howto about buiding `faiss` from source.

On `windows`, build will fail because `msvc` c++ compiler compatable [issue](https://github.com/facebookresearch/faiss/issues/2985), so I created a personal heck to make things work, https://github.com/yexiangyu/faiss/archive/refs/heads/v1.7.4-win.zip, if `windows` is not used, just clone `faiss` and checkout `v1.7.4` branch will just work.

Download the [zip](https://github.com/yexiangyu/faiss/archive/refs/heads/v1.7.4-win.zip), unzip then unzip it.

## `MacOS`
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

## `Linux`

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
## `Windows`
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



