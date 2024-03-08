#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("rc={}, message={}", .code, .message)]
    FaissError { code: i32, message: String },
    #[error("invalid index description")]
    InvalidIndexDescription,
    #[error("invalid combination name")]
    InvalidCombinationName,
    #[error("invalid index parameters")]
    InvalidIndexParameters,
    #[error("downcast index failure")]
    DowncastFailure,
}

impl From<i32> for Error {
    fn from(code: i32) -> Self {
        let message = unsafe {
            let c_str = std::ffi::CStr::from_ptr(faiss_next_sys::faiss_get_last_error());
            c_str.to_string_lossy().into_owned()
        };
        Error::FaissError { code, message }
    }
}

pub type Result<T> = std::result::Result<T, Error>;
