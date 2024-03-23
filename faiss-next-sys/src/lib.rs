#![allow(non_upper_case_globals, non_camel_case_types)]
#[cfg(all(
    target_os = "macos",
    target_arch = "aarch64",
    not(feature = "gpu"),
    not(feature = "bindgen")
))]
include!("macos-aarch64-cpu.rs");
