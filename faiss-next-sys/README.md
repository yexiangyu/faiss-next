# faiss-next-sys

[![Crates.io](https://img.shields.io/crates/v/faiss-next-sys.svg)](https://crates.io/crates/faiss-next-sys)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Raw FFI bindings to [Faiss](https://github.com/facebookresearch/faiss) C API.

This crate provides unsafe Rust bindings to the Faiss C library. For a safe, idiomatic API, use [faiss-next](https://crates.io/crates/faiss-next).

## Features

- Pre-generated bindings for common platforms
- Automatic Faiss version detection
- Multi-version binding support (1.14.x, 1.15.x, ...)
- Optional runtime binding generation via `bindgen`
- CUDA support (Linux and Windows)

## Supported Platforms

| OS | Architecture | CPU | CUDA |
|----|--------------|-----|------|
| macOS (Apple Silicon) | aarch64 (M1/M2/M3) | ✅ | ❌ |
| Linux | x86_64 | ✅ | ✅ |
| Windows | x86_64 | ✅ | ✅ |

**Legend:**
- ✅ Fully supported with pre-generated bindings
- ❌ Not supported

**Notes:**
- CUDA is supported on Linux x86_64 and Windows x86_64

## Getting Started

### Prerequisites

Install Faiss with C API enabled:

**macOS (Homebrew):**
```bash
brew install faiss
```

**Linux (from source):**
```bash
git clone https://github.com/facebookresearch/faiss.git
cd faiss
mkdir build && cd build
cmake -DFAISS_ENABLE_C_API=ON -DBUILD_SHARED_LIBS=ON ..
make -j$(nproc)
sudo make install
sudo ldconfig
```

**Windows:**

Build Faiss from source with C API enabled. A pre-configured build is available at:
```bash
git clone -b windows-build https://github.com/yexiangyu/faiss.git
cd faiss
mkdir build && cd build
cmake -A x64 -DFAISS_ENABLE_C_API=ON -DBUILD_SHARED_LIBS=ON ^
      -DFAISS_ENABLE_GPU=ON ^
      -DCMAKE_INSTALL_PREFIX=C:/tools/faiss ..
cmake --build . --config Release
cmake --install . --config Release
```

After installation, set environment variables:
```cmd
set FAISS_INCLUDE_DIR=C:\tools\faiss\include
set FAISS_LIB_DIR=C:\tools\faiss\lib
```

Alternatively, place Faiss in the default location:
- `C:\tools\faiss` (include: `C:\tools\faiss\include`, lib: `C:\tools\faiss\lib`)

Or other supported locations:
- `C:\faiss`
- `C:\Program Files\faiss`
- `C:\Program Files (x86)\faiss`

### Installation

This crate is typically used as a dependency of `faiss-next`. Add to `Cargo.toml`:

```toml
[dependencies]
faiss-next-sys = "0.6"
```

### Environment Variables

| Variable | Description |
|----------|-------------|
| `FAISS_INCLUDE_DIR` | Direct path to Faiss include directory (e.g., `C:\faiss\include`) |
| `FAISS_LIB_DIR` | Direct path to Faiss lib directory (e.g., `C:\faiss\lib`) |
| `FAISS_DIR` | Path to Faiss installation root (must contain `include/` and `lib/`) |
| `LD_LIBRARY_PATH` | Linux library search path |
| `DYLD_LIBRARY_PATH` | macOS library search path |
| `PATH` | Windows DLL search path |

**Priority order:**
1. `FAISS_INCLUDE_DIR` + `FAISS_LIB_DIR` (direct paths, recommended for Windows)
2. `FAISS_DIR` (installation root)
3. Platform-specific default paths

## Feature Flags

| Flag | Description |
|------|-------------|
| `cuda` | Enable CUDA support (Linux x86_64 and Windows x86_64 only) |
| `bindgen` | Generate bindings at compile time |

**Platform restrictions:**
- `cuda` feature is only available on Linux x86_64 and Windows x86_64
- Using `cuda` feature on macOS will result in a compile error

**Note:** The `bindgen` feature requires LLVM/Clang to be installed:
- **Linux:** `sudo apt install llvm-dev libclang-dev` (Ubuntu/Debian)
- **macOS:** `brew install llvm`
- **Windows:** Install LLVM from https://llvm.org/builds/ and ensure `libclang` is in your PATH

## Version Detection

The build script automatically detects the installed Faiss version by parsing `include/faiss/Index.h`:

```c
#define FAISS_VERSION_MAJOR 1
#define FAISS_VERSION_MINOR 14
#define FAISS_VERSION_PATCH 0
```

### Version Compatibility

- **Minimum version**: 1.14.0
- **Loose mode**: Versions newer than tested versions work with a compile warning

When a newer Faiss version is detected, the build system:
1. Emits a warning about compatibility
2. Uses the closest available binding version
3. Attempts to compile and run

## Binding Generation

### Pre-generated Bindings

Bindings are pre-generated and committed to the repository:

```text
src/bindings/
├── mod.rs              # Version selection
└── v1_14/
    ├── mod.rs          # Platform selection
    ├── macos_aarch64.rs
    ├── linux_x86_64.rs
    ├── linux_x86_64_cuda.rs
    ├── windows_x86_64.rs
    └── windows_x86_64_cuda.rs
```

### Generate New Bindings

To generate bindings for your platform (e.g., when using a custom Faiss version):

```bash
cargo build --features bindgen
```

## API Structure

```rust
// Types
pub type idx_t = i64;

// Index operations
pub fn faiss_IndexFlat_new(...) -> *mut FaissIndexFlat;
pub fn faiss_Index_add(...);
pub fn faiss_Index_search(...);

// IVF operations
pub fn faiss_IndexIVF_new(...);
pub fn faiss_IndexIVF_set_nprobe(...);

// Distance computations
pub fn faiss_pairwise_L2sqr(...);
pub fn faiss_fvec_inner_products_ny(...);

// ... and more
```

## Linking

The crate links against:
- `libfaiss` - Core Faiss library
- `libfaiss_c` - C API wrapper

Ensure these libraries are in your library search path.

## Documentation

- [Faiss C API Reference](https://github.com/facebookresearch/faiss/blob/main/c_api/faiss_c.h)
- [faiss-next Documentation](https://docs.rs/faiss-next)

## License

MIT License