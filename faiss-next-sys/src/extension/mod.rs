use std::os::raw::c_char;

#[cxx::bridge]
mod ffi {
    unsafe extern "C++" {
        include!("faiss-next-sys/src/extension/index_binary_factory.hpp");

        type FaissIndexBinary;
        type Char;

        unsafe fn faiss_index_binary_factory(
            d: i32,
            description: *const Char,
            index: *mut *mut FaissIndexBinary,
        ) -> i32;

    }
}

pub fn faiss_index_binary_factory(
    d: i32,
    description: *const c_char,
    index: *mut *mut super::FaissIndexBinary,
) -> i32 {
    unsafe { ffi::faiss_index_binary_factory(d, description as *const _, index as *mut *mut _) }
}
