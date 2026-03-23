use std::fmt;

pub type IdxRepr = i64;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Idx(pub IdxRepr);

impl Idx {
    pub const NONE: Self = Idx(-1);

    #[inline]
    pub fn new(idx: u64) -> Self {
        Idx(idx as IdxRepr)
    }

    #[inline]
    pub fn is_none(self) -> bool {
        self.0 < 0
    }

    #[inline]
    pub fn is_some(self) -> bool {
        self.0 >= 0
    }

    #[inline]
    pub fn get(self) -> Option<u64> {
        if self.0 < 0 {
            None
        } else {
            Some(self.0 as u64)
        }
    }

    #[inline]
    pub fn get_unchecked(self) -> u64 {
        self.0 as u64
    }

    #[inline]
    pub fn as_repr(self) -> IdxRepr {
        self.0
    }
}

impl Default for Idx {
    fn default() -> Self {
        Self::NONE
    }
}

impl From<u64> for Idx {
    fn from(idx: u64) -> Self {
        Self::new(idx)
    }
}

impl From<Option<u64>> for Idx {
    fn from(opt: Option<u64>) -> Self {
        match opt {
            Some(idx) => Self::new(idx),
            None => Self::NONE,
        }
    }
}

impl From<Idx> for Option<u64> {
    fn from(idx: Idx) -> Self {
        idx.get()
    }
}

impl fmt::Display for Idx {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.get() {
            Some(idx) => write!(f, "Idx({})", idx),
            None => write!(f, "Idx(None)"),
        }
    }
}
