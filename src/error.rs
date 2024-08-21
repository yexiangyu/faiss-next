use faiss_next_sys as ffi;
use std::ffi::CStr;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("code={}, message={}", .code, .message)]
    FaissInner { code: i32, message: String },
    #[error("null filename")]
    NullFilename(#[from] std::ffi::NulError),
}

pub type Result<T> = std::result::Result<T, Error>;

const OK: i32 = ffi::FaissErrorCode::OK as i32;

pub fn faiss_rc(code: i32) -> Result<()> {
    match code {
        OK => Ok(()),
        _ => {
            let c_str = unsafe { CStr::from_ptr(ffi::faiss_get_last_error()) };
            Err(Error::FaissInner {
                code,
                message: c_str.to_str().unwrap_or("unknown error message").into(),
            })
        }
    }
}
