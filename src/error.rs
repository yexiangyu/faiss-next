use std::ffi::{CStr, NulError};
use std::io;

use faiss_next_sys;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Faiss native error (code={code}): {message}")]
    Native { code: i32, message: String },

    #[error("Invalid index description: {0}")]
    InvalidDescription(String),

    #[error("Index not trained")]
    NotTrained,

    #[error("Index is empty")]
    EmptyIndex,

    #[error("Invalid dimension: expected {expected}, got {actual}")]
    InvalidDimension { expected: usize, actual: usize },

    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    #[error("Invalid cast to {target}: {reason}")]
    InvalidCast {
        target: &'static str,
        reason: String,
    },

    #[error("Null pointer encountered")]
    NullPointer,

    #[error("Index does not support operation: {0}")]
    UnsupportedOperation(&'static str),

    #[error("IO error: {0}")]
    Io(#[from] io::Error),

    #[error("UTF-8 error: {0}")]
    Utf8(#[from] NulError),
}

pub type Result<T> = std::result::Result<T, Error>;

#[inline]
pub(crate) fn check_return_code(code: i32) -> Result<()> {
    if code == faiss_next_sys::FAISS_OK {
        return Ok(());
    }

    let message = unsafe {
        let ptr = faiss_next_sys::faiss_get_last_error();
        if ptr.is_null() {
            "unknown error".to_string()
        } else {
            CStr::from_ptr(ptr)
                .to_str()
                .unwrap_or("invalid error message")
                .to_string()
        }
    };

    Err(Error::Native { code, message })
}

impl Error {
    pub fn native(code: i32, message: impl Into<String>) -> Self {
        Error::Native {
            code,
            message: message.into(),
        }
    }

    pub fn invalid_cast(target: &'static str, reason: impl Into<String>) -> Self {
        Error::InvalidCast {
            target,
            reason: reason.into(),
        }
    }

    pub fn unsupported(operation: &'static str) -> Self {
        Error::UnsupportedOperation(operation)
    }
}
