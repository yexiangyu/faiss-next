macro_rules! rc {
    ($blk: block) => {{
        let r = unsafe { $blk };
        match r {
            0 => Ok(()),
            _ => Err(crate::error::Error::from(r)),
        }
    }};
}

pub(crate) use rc;
