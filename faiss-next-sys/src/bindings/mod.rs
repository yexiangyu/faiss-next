#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]
#![allow(improper_ctypes)]

#[cfg(faiss_binding = "v1_14")]
mod v1_14;

#[cfg(faiss_binding = "v1_14")]
pub use v1_14::*;

#[cfg(faiss_binding = "v1_15")]
mod v1_15;

#[cfg(faiss_binding = "v1_15")]
pub use v1_15::*;