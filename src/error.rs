use faiss_next_sys as sys;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("invalid parameter spance name")]
    InvalidParameterSpaceName,
    #[error("invalid parameter range name")]
    InvalidParameterRangeName,
    #[error("failed to cast type")]
    CastFailed,
    #[error("invalid description")]
    InvalidDescription,
    #[error("faiss internal error code={}, message={}", .code, .msg)]
    Fassis { code: i32, msg: String },
}

pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    pub fn from_code(code: i32) -> Self {
        let msg = unsafe {
            let c_str = sys::faiss_get_last_error();
            let msg = std::ffi::CStr::from_ptr(c_str)
                .to_str()
                .unwrap_or("unknown error with utf-8 decode failure")
                .to_string();
            msg
        };
        Error::Fassis { code, msg }
    }
}
