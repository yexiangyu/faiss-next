use std::ffi::CStr;

use crate::bindings;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("faiss error (code={code}): {message}")]
    Faiss { code: i32, message: String },
    #[error("null pointer encountered")]
    NullPointer,
    #[error("index not trained")]
    NotTrained,
    #[error("index is empty")]
    EmptyIndex,
    #[error("invalid dimension: expected {expected}, got {actual}")]
    InvalidDimension { expected: usize, actual: usize },
    #[error("invalid parameter: {0}")]
    InvalidParameter(String),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("UTF-8 error: {0}")]
    Utf8(#[from] std::ffi::NulError),
}

pub type Result<T> = std::result::Result<T, Error>;

const FAISS_OK: i32 = 0;

pub fn check_return_code(code: i32) -> Result<()> {
    if code == FAISS_OK {
        return Ok(());
    }

    let message = unsafe {
        let ptr = bindings::faiss_get_last_error();
        if ptr.is_null() {
            "unknown error".to_string()
        } else {
            CStr::from_ptr(ptr)
                .to_str()
                .unwrap_or("invalid error message")
                .to_string()
        }
    };

    Err(Error::Faiss { code, message })
}
