use burn::backend::wgpu::AutoGraphicsApi;
use burn::backend::{Autodiff, Wgpu};
use burn::module::AutodiffModule;
use burn::nn::loss::{MseLoss, Reduction};
use burn::optim::adaptor::OptimizerAdaptor;
use burn::optim::{AdamW, AdamWConfig, GradientsParams, Optimizer};
use burn::tensor::backend::AutodiffBackend;
use burn::{optim::AdamConfig, prelude::*};

use crate::train::model::*;
use crate::train::data::*;
use shared_types::*;

#[derive(Config)]
struct TheTrainingConfig {
    #[config(default = 2)]
    pub num_epochs: usize,
    #[config(default = 64)]
    pub batch_size: usize,
    #[config(default = 12)]
    pub num_workers: usize,
    #[config(default = 42)]
    pub seed: u64,
    #[config(default = 1e-4)]
    pub lr: f64,
    pub model: TheModelConfig,
    pub optimizer: AdamWConfig,
}

pub struct Trainer<B: AutodiffBackend> {
    device: B::Device,
    config: TheTrainingConfig,
    dataset: TheDataset,
    batcher: TheBatcher<B>,
    model: TheModel<B>,
    optimizer: OptimizerAdaptor<AdamW<B::InnerBackend>, TheModel<B>, B>,
    loss: MseLoss<B>,
}

pub type TheBackend = Wgpu<AutoGraphicsApi, f32, i32>;
pub type TheAutodiffBackend = Autodiff<TheBackend>;
pub fn make_trainer() -> Trainer<TheAutodiffBackend> {
    let device = burn::backend::wgpu::WgpuDevice::default();
    Trainer::new(device)
}

impl<B: AutodiffBackend> Trainer<B> {
    pub fn new(device: B::Device) -> Self {

        let model_config = TheModelConfig::new(NUM_FEATURES, SERIES_LENGTH, MODEL_OUTPUT_WIDTH);
        let dataset = TheDataset::train();
        let batcher = TheBatcher::<B>::new(device.clone());
        let model = model_config.init::<B>(&device);
        let config = TheTrainingConfig::new(model_config, AdamWConfig::new());
        let loss = MseLoss::new();
        let optimizer = AdamWConfig::new().init::<B,TheModel<B>>();

        Self { device, config, dataset, batcher, model, loss, optimizer }
    }

    // pub fn convert(&self, quotes: Vec<QuoteEvent>, label: Label) {
    //     println!("{:?}", label);
    // }

    pub fn train_batch(&mut self, input: Tensor<B,3>, expected: Tensor<B,2>) {
        let output = self.model.forward(input.clone());
        let loss = self.loss.forward(output, expected.clone(), Reduction::Mean);
        println!("Initial loss: {}", loss.to_data());
        let grads = GradientsParams::from_grads(loss.backward(), &self.model);
        self.model = self.optimizer.step(self.config.lr, self.model.clone(), grads);


        let output2 = self.model.forward(input);
        let loss2 = self.loss.forward(output2, expected, Reduction::Mean);
        println!("Initial loss: {}", loss2.to_data());
    }

    pub fn train_1(&mut self, input: [[f32; NUM_FEATURES]; SERIES_LENGTH], expected: [f32; MODEL_OUTPUT_WIDTH]) {
        let x = Tensor::from_floats(input, &self.device).unsqueeze();
        let y = Tensor::from_floats(expected, &self.device).unsqueeze();
        self.train_batch(x, y);
    }
}

