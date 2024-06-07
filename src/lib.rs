pub mod data;
pub mod train;
pub mod convert;
pub mod inferer;
pub mod output;
mod model_persist;
mod events_window;

use burn::backend::{wgpu::AutoGraphicsApi, Autodiff, Wgpu};

pub type TheBackend = Wgpu<AutoGraphicsApi, f32, i32>;
pub type TheAutodiffBackend = Autodiff<TheBackend>;
