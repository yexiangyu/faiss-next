use std::env;
use std::path::{Path, PathBuf};

fn main() {
    // Load .env file if it exists
    dotenv::dotenv().ok();

    // Tell cargo to invalidate the built crate whenever wrapper.h changes
    println!("cargo:rerun-if-changed=wrapper.h");

    // Get FAISS include and library paths from environment or use defaults
    let faiss_inc_path = env::var("FAISS_INC_PATH")
        .ok()
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| {
            // Auto-detect on macOS
            if cfg!(target_os = "macos") {
                let homebrew_path = "/opt/homebrew/opt/faiss/include/faiss/c_api";
                if Path::new(homebrew_path).exists() {
                    println!("cargo:warning=Using Homebrew FAISS include path: {}", homebrew_path);
                    return homebrew_path.to_string();
                }
            }
            panic!("FAISS_INC_PATH environment variable not set and could not auto-detect FAISS installation");
        });

    let faiss_lib_path = env::var("FAISS_LIB_PATH")
        .ok()
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| {
            // Auto-detect on macOS
            if cfg!(target_os = "macos") {
                let homebrew_path = "/opt/homebrew/opt/faiss/lib";
                if Path::new(homebrew_path).exists() {
                    println!("cargo:warning=Using Homebrew FAISS library path: {}", homebrew_path);
                    return homebrew_path.to_string();
                }
            }
            panic!("FAISS_LIB_PATH environment variable not set and could not auto-detect FAISS installation");
        });

    // Tell cargo to link against faiss library
    println!("cargo:rustc-link-search=native={}", faiss_lib_path);
    println!("cargo:rustc-link-lib=faiss");
    println!("cargo:rustc-link-lib=faiss_c");

    // Also need the parent include directory for C++ headers
    let faiss_base_inc = PathBuf::from(&faiss_inc_path)
        .parent()
        .and_then(|p| p.parent())
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_else(|| "/opt/homebrew/opt/faiss/include".to_string());

    // Generate bindings
    let bindings = bindgen::Builder::default()
        // Input header file
        .header("wrapper.h")
        // Add FAISS C API include path
        .clang_arg(format!("-I{}", faiss_inc_path))
        // Add FAISS base include path for C++ headers
        .clang_arg(format!("-I{}", faiss_base_inc))
        // Enable C++ mode for parsing C++ headers
        .clang_arg("-x")
        .clang_arg("c++")
        .clang_arg("-std=c++11")
        // Tell cargo to invalidate the built crate whenever any of the
        // included header files changed
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        // Generate bindings
        .generate()
        .expect("Unable to generate bindings");

    // Write the bindings to the $OUT_DIR/bindings.rs file
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    let bindings_path = out_path.join("bindings.rs");
    println!("cargo:warning=Writing bindings to: {}", bindings_path.display());
    bindings
        .write_to_file(&bindings_path)
        .expect("Couldn't write bindings!");
}
