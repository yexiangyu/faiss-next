use faiss_next_sys as sys;
use std::ffi::CStr;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("faiss error code={}, message={}", .code, .message)]
    Faiss { code: i32, message: String },
    #[error("invalid description={}", .desc)]
    InvalidDescription { desc: String },
    #[error("invalid path={}", .path)]
    InvalidPath { path: String },
    #[error("not standard layout")]
    NotStandardLayout,
    #[error("invalid devices=[]")]
    InvalidGpuDevices,
}

impl From<i32> for Error {
    fn from(code: i32) -> Self {
        let message = unsafe {
            let c_str = sys::faiss_get_last_error();
            match c_str.is_null() {
                true => "unknown error".into(),
                false => {
                    let slice = CStr::from_ptr(c_str).to_bytes();
                    String::from_utf8_lossy(slice).into_owned()
                }
            }
        };
        Error::Faiss { code, message }
    }
}

pub type Result<T> = std::result::Result<T, Error>;
