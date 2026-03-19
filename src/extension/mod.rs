use std::ffi::c_char;

#[cxx::bridge]
mod ffi {
    unsafe extern "C++" {
        include!("faiss-next/src/extension/index_binary_factory.hpp");

        type FaissIndexBinary;

        unsafe fn faiss_index_binary_factory(
            d: i32,
            description: *const c_char,
            index: *mut *mut FaissIndexBinary,
        ) -> i32;
    }
}

pub fn faiss_index_binary_factory(
    d: i32,
    description: *const c_char,
    index: *mut *mut crate::bindings::FaissIndexBinary,
) -> i32 {
    unsafe { ffi::faiss_index_binary_factory(d, description, index as *mut *mut _) }
}

#[cfg(feature = "cuda")]
pub mod cuda;
