#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]
#![allow(improper_ctypes)]

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
include!("macos_aarch64.rs");

#[cfg(all(target_os = "linux", target_arch = "x86_64", not(feature = "cuda")))]
include!("linux_x86_64.rs");

#[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
include!("linux_x86_64_cuda.rs");

#[cfg(all(target_os = "windows", target_arch = "x86_64", not(feature = "cuda")))]
include!("windows_x86_64.rs");

#[cfg(all(target_os = "windows", target_arch = "x86_64", feature = "cuda"))]
include!("windows_x86_64_cuda.rs");
