use crate::ffi;
use std::ffi::CStr;
use std::fmt;

/// FAISS error codes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorCode {
    /// FAISS-specific exception
    FaissException = -2,
    /// Standard C++ library exception
    StdException = -4,
    /// Unknown error
    Unknown,
}

impl From<i32> for ErrorCode {
    fn from(code: i32) -> Self {
        match code {
            -2 => ErrorCode::FaissException,
            -4 => ErrorCode::StdException,
            _ => ErrorCode::Unknown,
        }
    }
}

/// FAISS error type
#[derive(Debug, Clone)]
pub struct FaissError {
    /// Error code
    pub code: ErrorCode,
    /// Error message
    pub message: String,
}

impl FaissError {
    /// Create a new FaissError with the given code and message
    pub fn new(code: ErrorCode, message: String) -> Self {
        Self { code, message }
    }

    /// Create a FaissError from a return code
    ///
    /// If the return code indicates an error, retrieves the last error message from FAISS
    pub fn from_code(code: i32) -> Option<Self> {
        if code != 0 {
            let message = get_last_error();
            Some(Self {
                code: ErrorCode::from(code),
                message,
            })
        } else {
            None
        }
    }

    /// Create a FaissError with a custom message
    pub fn with_message(message: impl Into<String>) -> Self {
        Self {
            code: ErrorCode::Unknown,
            message: message.into(),
        }
    }
}

impl fmt::Display for FaissError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "FAISS error ({:?}): {}", self.code, self.message)
    }
}

impl std::error::Error for FaissError {}

/// Get the last error message from FAISS
pub fn get_last_error() -> String {
    unsafe {
        let err_ptr = ffi::faiss_get_last_error();
        if err_ptr.is_null() {
            return "Unknown FAISS error".to_string();
        }

        CStr::from_ptr(err_ptr)
            .to_string_lossy()
            .into_owned()
    }
}

/// Check a FAISS return code and return an error if it's non-zero
pub fn check_error(code: i32) -> Result<(), FaissError> {
    if let Some(err) = FaissError::from_code(code) {
        Err(err)
    } else {
        Ok(())
    }
}

/// Helper macro to check FAISS return codes and convert to Result
#[macro_export]
macro_rules! faiss_try {
    ($expr:expr) => {{
        let code = $expr;
        $crate::error::check_error(code)?;
    }};
}

/// Helper macro to check FAISS return codes and bail with context
#[macro_export]
macro_rules! faiss_try_with {
    ($expr:expr, $context:expr) => {{
        let code = $expr;
        if let Some(err) = $crate::error::FaissError::from_code(code) {
            anyhow::bail!("{}: {}", $context, err.message);
        }
    }};
}
