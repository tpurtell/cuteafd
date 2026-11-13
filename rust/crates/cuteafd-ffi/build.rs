use std::env;
use std::path::PathBuf;
use std::process::Command;

fn run(command: &mut Command) {
    let status = command.status().expect("run native test fixture build command");
    assert!(status.success(), "native test fixture build failed: {command:?}");
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    // Only dev-dependencies enable this feature; ordinary serving builds do no
    // fixture work and need neither Python nor a C/C++ compiler for this crate.
    if env::var_os("CARGO_FEATURE_TEST_SUPPORT").is_none() {
        return;
    }
    let manifest = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let repo = manifest.join("../../..");
    let out = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    for input in [
        "rust/crates/cuteafd-ffi/tests/fixtures/native_library_lifetime.c",
        "rust/crates/cuteafd-ffi/tests/fixtures/native_library_kernels.cc",
        "native/shared/cuda/expert_pack_layout.h",
        "native/shared/cuda/fp8_load_quant.h",
        "native/shared/cuda/load_scale_gather.h",
        "native/shared/include/memory_failure.h",
        "native/shared/include/cuteafd_memory_kind.h",
        "python/tools/lib/mxfp4_layouts.py",
        "scripts/build/mxfp4-layouts.json",
        "scripts/build/assert-build-filesystem.py",
    ] {
        println!("cargo:rerun-if-changed={}", repo.join(input).display());
    }
    run(Command::new("python3")
        .arg(repo.join("scripts/build/assert-build-filesystem.py")).arg(&out));
    run(Command::new("python3")
        .arg(repo.join("python/tools/lib/mxfp4_layouts.py"))
        .arg("--header").arg(out.join("cuteafd_mxfp4_layouts.h")));
    // Cargo's caller already holds the compile slot. Never acquire a nested
    // build slot here, or spawn a compiler from inside a timed test body.
    let compiler = |program: &str| {
        let mut command = Command::new("nice");
        command.args(["-n", "19", program]).env("TMPDIR", &out);
        command
    };
    let fixtures = manifest.join("tests/fixtures");
    run(compiler("cc")
        .args(["-c", "-fPIC", "-O0", "-Wall", "-Wextra", "-Werror"])
        .arg(fixtures.join("native_library_lifetime.c"))
        .arg("-I").arg(&out).arg("-o").arg(out.join("fixture.o")));
    run(compiler("c++")
        .args(["-std=c++17", "-c", "-fPIC", "-O1", "-Wall", "-Wextra", "-Werror"])
        .arg(fixtures.join("native_library_kernels.cc"))
        .arg("-o").arg(out.join("kernels.o")));
    run(compiler("c++").args(["-shared", "-fPIC"])
        .arg(out.join("fixture.o")).arg(out.join("kernels.o"))
        .arg("-o").arg(out.join("liblifetime_fixture.so")));
}
