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
- CUDA support (Linux only)

## Supported Platforms

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
- For unsupported platforms, enable `bindgen` feature

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

### Installation

This crate is typically used as a dependency of `faiss-next`. Add to `Cargo.toml`:

```toml
[dependencies]
faiss-next-sys = "0.6"
```

### Environment Variables

| Variable | Description |
|----------|-------------|
| `FAISS_DIR` | Path to Faiss installation (must contain `include/` and `lib/`) |
| `LD_LIBRARY_PATH` | Linux library search path |
| `DYLD_LIBRARY_PATH` | macOS library search path |

## Feature Flags

| Flag | Description |
|------|-------------|
| `cuda` | Enable CUDA support (Linux only) |
| `bindgen` | Generate bindings at compile time |

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
    ├── macos_aarch64_cuda.rs
    ├── linux_x86_64.rs
    └── linux_x86_64_cuda.rs
```

### Generate New Bindings

To generate bindings for your platform:

```bash
cargo build --features bindgen
```

This is useful when:
- Using an unsupported platform
- Using a custom Faiss build
- Adding support for a new Faiss version

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