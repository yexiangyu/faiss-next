use crate::error::FaissError;
use crate::ffi;
use crate::index::{IndexOwned, IndexTrait};
use anyhow::Result;
use std::ffi::CString;
use std::io::{Read, Write};
use std::os::raw::c_void;

/// FAISS I/O Reader
///
/// Wraps a Rust Read trait object to work with FAISS I/O functions
pub struct IOReader {
    inner: *mut ffi::FaissCustomIOReader,
    // Keep the boxed reader alive
    _reader: Box<Box<dyn Read>>,
}

impl IOReader {
    /// Create a new IOReader from a Rust Read trait object
    ///
    /// # Arguments
    /// * `reader` - Any type implementing std::io::Read
    pub fn new<R: Read + 'static>(reader: R) -> Result<Self> {
        // Box the reader twice: once for trait object, once for stable pointer
        let boxed_reader: Box<Box<dyn Read>> = Box::new(Box::new(reader));
        let reader_ptr = Box::into_raw(boxed_reader);

        unsafe {
            let mut io_reader = std::ptr::null_mut();
            let ret = ffi::faiss_CustomIOReader_new(&mut io_reader, Some(read_callback));

            if let Some(err) = FaissError::from_code(ret) {
                // Clean up on error
                drop(Box::from_raw(reader_ptr));
                return Err(err.into());
            }

            // Store the reader pointer in the user data (if FAISS supports it)
            // For now, we'll use a global or thread-local storage approach
            CURRENT_READER.with(|cell| {
                cell.borrow_mut().replace(reader_ptr as *mut c_void);
            });

            Ok(Self {
                inner: io_reader,
                _reader: Box::from_raw(reader_ptr),
            })
        }
    }

    /// Get the raw pointer for use with FAISS functions
    pub fn as_ptr(&self) -> *mut ffi::FaissIOReader {
        self.inner as *mut ffi::FaissIOReader
    }
}

impl Drop for IOReader {
    fn drop(&mut self) {
        if !self.inner.is_null() {
            unsafe {
                ffi::faiss_CustomIOReader_free(self.inner);
            }
        }
        CURRENT_READER.with(|cell| {
            cell.borrow_mut().take();
        });
    }
}

// Thread-local storage for the current reader/writer
thread_local! {
    static CURRENT_READER: std::cell::RefCell<Option<*mut c_void>> = const { std::cell::RefCell::new(None) };
    static CURRENT_WRITER: std::cell::RefCell<Option<*mut c_void>> = const { std::cell::RefCell::new(None) };
}

/// Callback function for reading data
unsafe extern "C" fn read_callback(
    ptr: *mut c_void,
    size: usize,
    nitems: usize,
) -> usize {
    unsafe {
        CURRENT_READER.with(|cell| {
            let reader_ptr = match cell.borrow().as_ref() {
                Some(&p) => p,
                None => return 0,
            };

            let reader = &mut *(reader_ptr as *mut Box<dyn Read>);
            let total_bytes = size * nitems;
            let buffer = std::slice::from_raw_parts_mut(ptr as *mut u8, total_bytes);

            match reader.read(buffer) {
                Ok(n) => n / size, // Return number of items read
                Err(_) => 0,
            }
        })
    }
}

/// Callback function for writing data
unsafe extern "C" fn write_callback(
    ptr: *const c_void,
    size: usize,
    nitems: usize,
) -> usize {
    unsafe {
        CURRENT_WRITER.with(|cell| {
            let writer_ptr = match cell.borrow().as_ref() {
                Some(&p) => p,
                None => return 0,
            };

            let writer = &mut *(writer_ptr as *mut Box<dyn Write>);
            let total_bytes = size * nitems;
            let buffer = std::slice::from_raw_parts(ptr as *const u8, total_bytes);

            match writer.write_all(buffer) {
                Ok(()) => nitems, // Return number of items written
                Err(_) => 0,
            }
        })
    }
}

/// FAISS I/O Writer
///
/// Wraps a Rust Write trait object to work with FAISS I/O functions
pub struct IOWriter {
    inner: *mut ffi::FaissCustomIOWriter,
    // Keep the boxed writer alive
    _writer: Box<Box<dyn Write>>,
}

impl IOWriter {
    /// Create a new IOWriter from a Rust Write trait object
    ///
    /// # Arguments
    /// * `writer` - Any type implementing std::io::Write
    pub fn new<W: Write + 'static>(writer: W) -> Result<Self> {
        // Box the writer twice: once for trait object, once for stable pointer
        let boxed_writer: Box<Box<dyn Write>> = Box::new(Box::new(writer));
        let writer_ptr = Box::into_raw(boxed_writer);

        unsafe {
            let mut io_writer = std::ptr::null_mut();
            let ret = ffi::faiss_CustomIOWriter_new(&mut io_writer, Some(write_callback));

            if let Some(err) = FaissError::from_code(ret) {
                // Clean up on error
                drop(Box::from_raw(writer_ptr));
                return Err(err.into());
            }

            // Store the writer pointer
            CURRENT_WRITER.with(|cell| {
                cell.borrow_mut().replace(writer_ptr as *mut c_void);
            });

            Ok(Self {
                inner: io_writer,
                _writer: Box::from_raw(writer_ptr),
            })
        }
    }

    /// Get the raw pointer for use with FAISS functions
    pub fn as_ptr(&self) -> *mut ffi::FaissIOWriter {
        self.inner as *mut ffi::FaissIOWriter
    }
}

impl Drop for IOWriter {
    fn drop(&mut self) {
        if !self.inner.is_null() {
            unsafe {
                ffi::faiss_CustomIOWriter_free(self.inner);
            }
        }
        CURRENT_WRITER.with(|cell| {
            cell.borrow_mut().take();
        });
    }
}

/// Write an index to a file
///
/// # Arguments
/// * `index` - The index to write (any type implementing IndexTrait)
/// * `filename` - Path to the output file
pub fn write_index(index: &impl IndexTrait, filename: &str) -> Result<()> {
    let c_filename = CString::new(filename)?;
    unsafe {
        let ret = ffi::faiss_write_index_fname(index.inner_ptr(), c_filename.as_ptr());
        if let Some(err) = FaissError::from_code(ret) {
            return Err(err.into());
        }
    }
    Ok(())
}

/// Read an index from a file
///
/// # Arguments
/// * `filename` - Path to the input file
pub fn read_index(filename: &str) -> Result<IndexOwned> {
    let c_filename = CString::new(filename)?;
    unsafe {
        let mut index_ptr = std::ptr::null_mut();
        let ret = ffi::faiss_read_index_fname(c_filename.as_ptr(), 0, &mut index_ptr);
        if let Some(err) = FaissError::from_code(ret) {
            return Err(err.into());
        }
        IndexOwned::from_raw(index_ptr)
    }
}

/// Write an index using a custom writer
///
/// # Arguments
/// * `index` - The index to write (any type implementing IndexTrait)
/// * `writer` - Any type implementing std::io::Write
pub fn write_index_custom<W: Write + 'static>(index: &impl IndexTrait, writer: W) -> Result<()> {
    let io_writer = IOWriter::new(writer)?;
    unsafe {
        let ret = ffi::faiss_write_index_custom(index.inner_ptr(), io_writer.as_ptr(), 0);
        if let Some(err) = FaissError::from_code(ret) {
            return Err(err.into());
        }
    }
    Ok(())
}

/// Read an index using a custom reader
///
/// # Arguments
/// * `reader` - Any type implementing std::io::Read
pub fn read_index_custom<R: Read + 'static>(reader: R) -> Result<IndexOwned> {
    let io_reader = IOReader::new(reader)?;
    unsafe {
        let mut index_ptr = std::ptr::null_mut();
        let ret = ffi::faiss_read_index_custom(io_reader.as_ptr(), 0, &mut index_ptr);
        if let Some(err) = FaissError::from_code(ret) {
            return Err(err.into());
        }
        IndexOwned::from_raw(index_ptr)
    }
}
