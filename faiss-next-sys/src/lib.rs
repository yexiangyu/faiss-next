// #![allow(non_camel_case_types)]
// #[cfg(all(target_os = "macos", target_arch = "aarch64", not(feature = "cuda")))]
// include!("macos/aarch64.rs");
// #[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "cuda"))]
// include!("macos/aarch64_cuda.rs");

// #[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
// include!("linux/x86_64_cuda.rs");
// #[cfg(all(target_os = "linux", target_arch = "x86_64", not(feature = "cuda")))]
// include!("linux/x86_64.rs");

// pub mod extension;

#![allow(non_upper_case_globals, non_camel_case_types)]

use std::ffi::CStr;
#[cfg(all(target_os = "macos", target_arch = "aarch64", not(feature = "cuda")))]
include!("macos/aarch64/cpu.rs");

#[derive(Debug, thiserror::Error)]
pub struct Error {
    pub code: i32,
    pub message: String,
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Error {}: {}", self.code, self.message)
    }
}

pub type Result<T> = std::result::Result<T, Error>;

pub const OK: i32 = FaissErrorCode::OK as i32;

pub fn rc(code: i32) -> Result<()> {
    match code {
        OK => Ok(()),
        _ => Err(Error {
            code,
            message: unsafe { CStr::from_ptr(faiss_get_last_error()) }
                .to_str()
                .unwrap_or("Unknown error")
                .into(),
        }),
    }
}

#[macro_export]
macro_rules! impl_drop {
    ($cls: ident, $drop: ident) => {
        impl Drop for $cls {
            fn drop(&mut self) {
                tracing::trace!(?self, "dropping");
                unsafe { faiss_next_sys::$drop(self.inner as _) };
            }
        }
    };
}

#[macro_export]
macro_rules! run {
    ($fun: ident) => {
        unsafe { faiss_next_sys::$fun() }
    };

    ($fun: ident, $($args: expr),+) => {
        unsafe { faiss_next_sys::$fun($($args),+) }
    };
}

#[macro_export]
macro_rules! ok {
    ($fun: ident) => {
        faiss_next_sys::rc(faiss_next_sys::run!($fun))
    };

    ($fun: ident, $($args: expr),+) => {
        faiss_next_sys::rc(faiss_next_sys::run!($fun, $($args),+))
    };
}

#[macro_export]
macro_rules! impl_getter {
    ($cls: ident, $getter: ident, $ffi_getter: ident, $rt: ty) => {
        impl $cls {
            pub fn $getter(&self) -> $rt {
                unsafe { faiss_next_sys::$ffi_getter(self.inner) as _ }
            }
        }
    };
}

#[macro_export]
macro_rules! impl_setter {
    ($cls: ident, $setter: ident, $ffi_setter: ident, $val: ident, $vt: ty) => {
        impl $cls {
            pub fn $setter(&mut self, $val: $vt) {
                unsafe { faiss_next_sys::$ffi_setter(self.inner, $val) }
            }
        }
    };
}
