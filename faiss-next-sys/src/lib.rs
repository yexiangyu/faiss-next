#![doc = include_str!("../README.md")]
#![allow(non_camel_case_types)]

#[cfg(all(not(feature = "gpu"), target_os = "macos"))]
include!("macos/bindings.rs");

#[cfg(all(not(feature = "gpu"), target_os = "windows"))]
include!("windows/bindings.rs");

#[cfg(all(feature = "gpu", target_os = "windows"))]
include!("windows/bindings_gpu.rs");

#[cfg(all(not(feature = "gpu"), target_os = "linux"))]
include!("linux/bindings.rs");

#[cfg(all(feature = "gpu", target_os = "linux"))]
include!("linux/bindings_gpu.rs");
