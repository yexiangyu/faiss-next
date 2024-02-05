use faiss_next_sys as sys;

/// Error
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("invalid index description {0}")]
    InvalidDescription(String),
    #[error("invalid index dimension")]
    InvalidDimension,
    #[error("invalid cluster number")]
    InvalidClusterNumber,
    #[error("gpu {0} not available")]
    GpuNotAvailable(i32),
    #[error("index already on gpu")]
    IndexOnGpu,
    #[error("index already on cpu")]
    IndexOnCpu,
    #[error("{0}")]
    NulErr(#[from] std::ffi::NulError),
    #[error("faiss error,code={},message={}", .code, .message)]
    Faiss { code: i32, message: String },
}

impl Error {
    pub fn from_rc(rc: i32) -> Self {
        match rc {
            0 => unimplemented!(),
            _ => {
                let message = unsafe { std::ffi::CStr::from_ptr(sys::faiss_get_last_error()) };
                let message = message
                    .to_str()
                    .unwrap_or("unknown error, failed to decode error message from bytes")
                    .to_string();
                Error::Faiss { code: rc, message }
            }
        }
    }
}

pub type Result<T> = std::result::Result<T, Error>;
