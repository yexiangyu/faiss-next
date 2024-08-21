#![allow(non_camel_case_types)]
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
include!("macos/aarch64.rs");
