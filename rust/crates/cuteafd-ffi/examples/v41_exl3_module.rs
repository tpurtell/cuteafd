//! Load trusted EXL3 modules together and inspect their independent native ABIs.
use anyhow::{ensure, Result};
use cuteafd_ffi::V41Exl3Kernel;
fn main() -> Result<()> {
    let mut paths: Vec<_> = std::env::args_os().skip(1).collect();
    if paths.first().is_some_and(|arg| arg == "--family") {
        ensure!(paths.len() >= 3, "--family needs a family and library");
        let geometry = match paths[1].to_str() {
            Some("v41") => cuteafd_core::ExpertGeometry::DEEPSEEK_V41,
            Some("kimik3") => cuteafd_core::ExpertGeometry::KIMI_K3,
            _ => anyhow::bail!("supported probe families: v41, kimik3"),
        };
        cuteafd_core::set_expert_geometry(geometry)
            .map_err(|current| anyhow::anyhow!("geometry already initialized: {current:?}"))?;
        paths.drain(..2);
    }
    ensure!(!paths.is_empty(), "usage: v41_exl3_module [--family v41|kimik3] LIBRARY [LIBRARY ...]");
    let mut kernels = Vec::new();
    for path in paths {
        let kernel = unsafe { V41Exl3Kernel::load(&path)? };
        println!("{}: {:?}", std::path::Path::new(&path).display(), kernel.info());
        unsafe {
            assert!(kernel.launch_core(&[], &[], std::ptr::null_mut()).is_err());
            assert!(kernel.launch_sum(&[], &[], std::ptr::null_mut()).is_err());
        }
        kernels.push(kernel);
    }
    println!("{} modules loaded concurrently; native initialization and argument-count rejection passed", kernels.len());
    Ok(())
}
