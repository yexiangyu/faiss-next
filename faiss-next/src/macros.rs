macro_rules! faiss_rc {
    ($blk: block) => {
        match unsafe { $blk } {
            0 => Ok(()),
            rc => Err($crate::error::Error::from_rc(rc)),
        }
    };
}

pub(crate) use faiss_rc;
