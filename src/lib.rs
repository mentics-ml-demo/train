pub mod data;
pub mod train;
pub mod convert;
pub mod inferer;
pub mod output;
mod model_persist;

use std::path::PathBuf;
use anyhow::Context;
use burn::backend::{wgpu::AutoGraphicsApi, Autodiff, Wgpu};

pub type TheBackend = Wgpu<AutoGraphicsApi, f32, i32>;
pub type TheAutodiffBackend = Autodiff<TheBackend>;

pub(crate) fn artifacts_dir() -> anyhow::Result<PathBuf> {
    let h = home::home_dir().with_context(|| "Could not get user home directory")?;
    let path = h.join("data").join("models").join("oml");
    std::fs::create_dir_all(&path)?;
    Ok(path)
}
