#[cfg(all(target_os = "linux", feature = "cuda"))]
mod index;
#[cfg(all(target_os = "linux", feature = "cuda"))]
mod resources;

#[cfg(all(target_os = "linux", feature = "cuda"))]
pub use index::GpuIndexImpl;
#[cfg(all(target_os = "linux", feature = "cuda"))]
pub use resources::GpuResources;

#[cfg(not(all(target_os = "linux", feature = "cuda")))]
compile_error!(
    "GPU support is only available on Linux with the 'cuda' feature enabled. \
     Use `cargo build --features cuda` on a Linux system with CUDA installed."
);
