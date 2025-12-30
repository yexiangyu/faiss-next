use faiss_next_sys as ffi;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{0}")]
    Faiss(#[from] ffi::Error),
    #[error("{0}")]
    NulError(#[from] std::ffi::NulError),
}

pub type Result<T> = std::result::Result<T, Error>;

// pub fn rc(code: i32) -> Result<()> {
//     match code {
//         OK => Ok(()),
//         _ => {
//             let c_str = unsafe { CStr::from_ptr(ffi::faiss_get_last_error()) };
//             Err(Error::Faiss {
//                 code,
//                 message: c_str.to_str().unwrap_or("unknown error message").into(),
//             })
//         }
//     }
// }
