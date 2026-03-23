#![doc = include_str!("../README.md")]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]
#![allow(improper_ctypes)]

mod version;
mod bindings;

pub use bindings::*;
pub use version::*;

pub const FAISS_OK: i32 = 0;