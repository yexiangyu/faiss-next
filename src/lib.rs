#![allow(non_camel_case_types)]

pub mod bindings;
pub mod error;
pub mod macros;
pub mod prelude;
pub mod traits;

pub mod extension;

pub mod index;
pub mod index_factory;
pub mod index_flat;

pub mod index_binary;
pub mod index_io;
pub mod index_ivf;
pub mod index_ivf_flat;
pub mod index_lsh;

pub mod clustering;
pub mod impl_aux_index_structure;
pub mod vector_transform;

#[cfg(feature = "cuda")]
pub mod cuda;

#[cfg(test)]
mod accuracy_tests;
