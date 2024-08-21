#![allow(non_camel_case_types)]
#[cfg(all(target_os = "macos", target_arch = "aarch64", not(feature = "cuda")))]
include!("macos/aarch64.rs");
#[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "cuda"))]
include!("macos/aarch64_cuda.rs");

#[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
include!("linux/x86_64_cuda.rs");
#[cfg(all(target_os = "linux", target_arch = "x86_64", not(feature = "cuda")))]
include!("linux/x86_64.rs");

pub mod extension;
