#![allow(unused)]
use bindgen::Builder;

fn main() {
    // ignore build when detect run docs build on docs.rs
    if std::env::var("DOCS_RS").is_ok() {
        return;
    }

    // linking faiss libraries
    println!("cargo:rustc-link-lib=faiss_c");
    println!("cargo:rustc-link-lib=faiss");
    println!("cargo:rustc-link-search=c:\\tools\\faiss\\lib");

    // generate bindings for faiss
    #[cfg(feature = "bindgen")]
    {
        println!("cargo:rerun-if-changed=faiss.h");

        // generated bindings filename
        let mut filename = "bindings.rs".to_string();
        #[cfg(feature = "gpu")]
        {
            filename = "bindings_gpu.rs".to_string();
        }

        // create builder
        let mut builder = Builder::default()
            .header("faiss.h")
            .default_enum_style(bindgen::EnumVariation::Rust {
                non_exhaustive: false,
            })
            .layout_tests(false)
            .allowlist_function("faiss_.*")
            .allowlist_type("idx_t|Faiss.*")
            .opaque_type("FILE")
            .wrap_unsafe_ops(true);

        // enable gpu
        #[cfg(feature = "gpu")]
        {
            builder = builder.clang_arg("-DFAISS_USE_GPU");

            //include cuda headers
            #[cfg(target_os = "linux")]
            {
                builder = builder.clang_arg("-I/usr/local/cuda/targets/x86_64-linux/include");
            }

            // need a windows machine to generate the code
            #[cfg(target_os = "windows")]
            {
                builder = builder.clang_arg(
                    "-IC:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include",
                );
            }
        }

        // add os folder
        let mut os_dir = Option::<String>::None;

        #[cfg(target_os = "linux")]
        {
            os_dir = Some("linux".to_string());
        }

        #[cfg(target_os = "macos")]
        {
            os_dir = Some("macos".to_string());
        }

        #[cfg(target_os = "windows")]
        {
            os_dir = Some("windows".to_string());
        }

        // include faiss headers if env detected
        if let Some(dir) = get_faiss_inc_dir() {
            builder = builder.clang_arg(format!("-I{}", dir));
        }

        // add faiss link dir if env detected
        if let Some(dir) = get_faiss_lib_dir() {
            println!("cargo:rustc-link-search={}", dir);
        }

        // generate bindings
        let bindings = builder.generate().expect("failed to build bindings");

        // write to *.rs file
        let out = std::path::PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap())
            .join("src")
            .join(os_dir.expect("os not supported"));

        bindings
            .write_to_file(out.join(filename))
            .expect("could not write bindings");
    }
}

fn get_faiss_lib_dir() -> Option<String> {
    if let Ok(dir) = std::env::var("FAISS_LIB_DIR") {
        return Some(dir);
    }

    if let Ok(dir) = std::env::var("FAISS_DIR") {
        return Some(format!("{}{}lib", dir, std::path::MAIN_SEPARATOR));
    }

    None
}

fn get_faiss_inc_dir() -> Option<String> {
    if let Ok(dir) = std::env::var("FAISS_INC_DIR") {
        return Some(dir);
    }

    if let Ok(dir) = std::env::var("FAISS_DIR") {
        return Some(format!("{}{}include", dir, std::path::MAIN_SEPARATOR));
    }

    None
}
