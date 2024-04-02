use faiss_next_sys as sys;

#[repr(u32)]
#[derive(Clone, Debug, Copy, PartialEq, Eq)]
pub enum IndicesOptions {
    CPU = 0,
    IVF = 1,
    Bit32 = 2,
    Bit64 = 3,
}

impl From<sys::FaissIndicesOptions> for IndicesOptions {
    fn from(value: sys::FaissIndicesOptions) -> Self {
        match value {
            sys::FaissIndicesOptions::INDICES_CPU => Self::CPU,
            sys::FaissIndicesOptions::INDICES_IVF => Self::IVF,
            sys::FaissIndicesOptions::INDICES_32_BIT => Self::Bit32,
            sys::FaissIndicesOptions::INDICES_64_BIT => Self::Bit64,
            _ => panic!("invalid indice options"),
        }
    }
}

impl From<IndicesOptions> for sys::FaissIndicesOptions {
    fn from(value: IndicesOptions) -> Self {
        match value {
            IndicesOptions::CPU => sys::FaissIndicesOptions::INDICES_CPU,
            IndicesOptions::IVF => sys::FaissIndicesOptions::INDICES_IVF,
            IndicesOptions::Bit32 => sys::FaissIndicesOptions::INDICES_32_BIT,
            IndicesOptions::Bit64 => sys::FaissIndicesOptions::INDICES_64_BIT,
        }
    }
}
