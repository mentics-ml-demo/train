use std::iter::zip;
use std::path::{Path, PathBuf};

use anyhow::Context;
use burn::backend::wgpu::AutoGraphicsApi;
use burn::backend::{Autodiff, Wgpu};
use burn::nn::loss::{MseLoss, Reduction};
use burn::optim::adaptor::OptimizerAdaptor;
use burn::optim::{AdamW, AdamWConfig, GradientsParams, Optimizer};
use burn::record::{CompactRecorder, Recorder};
use burn::tensor::backend::AutodiffBackend;
use burn::prelude::*;
use tabled::builder::Builder;
use tabled::settings::Style;

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
    pub learning_rate: f64,
    pub model: TheModelConfig,
    pub optimizer: AdamWConfig,
}

/// Regarding the Option model:
/// https://play.rust-lang.org/?version=stable&mode=debug&edition=2021&gist=2cfe1e7a3eaec1d6cf9ac41e9d4778fc
/// https://stackoverflow.com/questions/54293155/how-to-replace-the-value-of-a-struct-field-without-taking-ownership/54293313#54293313
pub struct Trainer<B: AutodiffBackend> {
    device: B::Device,
    config: TheTrainingConfig,
    model: Option<TheModel<B>>,
    optimizer: OptimizerAdaptor<AdamW<B::InnerBackend>, TheModel<B>, B>,
    loss: MseLoss<B>,
    artifact_dir: PathBuf,
}

// pub fn test() -> anyhow::Result<()> {
//     let t = make_trainer("blue")?;
//     t.train_1([[0f32; NUM_FEATURES]; SERIES_LENGTH], [0f32; MODEL_OUTPUT_WIDTH]);
//     t.train_1([[0f32; NUM_FEATURES]; SERIES_LENGTH], [0f32; MODEL_OUTPUT_WIDTH]);
//     Ok(())
// }

pub type TheBackend = Wgpu<AutoGraphicsApi, f32, i32>;
pub type TheAutodiffBackend = Autodiff<TheBackend>;
pub fn make_trainer<P: AsRef<Path>>(artifact_dir: P) -> anyhow::Result<Trainer<TheAutodiffBackend>> {
    let device = burn::backend::wgpu::WgpuDevice::default();
    Trainer::new(device, artifact_dir)
}

impl<B: AutodiffBackend> Trainer<B> {
    pub fn new<P: AsRef<Path>>(device: B::Device, artifact_dir: P) -> anyhow::Result<Self> {
        let base_path = artifact_dir.as_ref();
        let model_config = TheModelConfig::new(NUM_FEATURES, SERIES_SIZE, MODEL_OUTPUT_WIDTH);
        let dataset = TheDataset::train();
        let batcher = TheBatcher::<B>::new(device.clone());
        let mut model = model_config.init::<B>(&device);
        let config = TheTrainingConfig::new(model_config, AdamWConfig::new());
        let loss = MseLoss::new();
        let optimizer = AdamWConfig::new().init::<B,TheModel<B>>();

        let config_path = base_path.join("config.json");
        if !config_path.try_exists()? {
            // TODO: manage changes to the config
            std::fs::create_dir_all(base_path)?;
            config.save(&config_path)?;
            println!("Config saved: {:?}", config_path);
        }

        let model_path = base_path.join("model.mpk");
        if model_path.try_exists()? {
            println!("Loading model from {:?}", &model_path);
            let record = CompactRecorder::new().load(model_path, &device).with_context(|| "Error loading model")?;
            model = config.model.init::<B>(&device).load_record(record);
        }

        Ok(Self { device, config, model: Some(model), loss, optimizer, artifact_dir: base_path.to_path_buf() })
    }

    pub fn save_model(&mut self) -> anyhow::Result<()> {
        // TODO: save only occasionally? and coordinate with commiting offsets
        let path = self.artifact_dir.join("model");
        self.model.clone().save_file(&path, &CompactRecorder::new()).with_context(|| "Error saving model")?;
        println!("Trained model saved: {:?}", &path);
        Ok(())
    }

    /// # Shapes
    ///   - Input [batch_size, SERIES_LENGTH, NUM_FEATURES]
    ///   - Expected [batch_size, MODEL_OUTPUT_WIDTH]
    ///   - Output [batch_size] of losses
    pub fn train_batch(&mut self, input: impl ToTensor<B,3>, expected: impl ToTensor<B,2>) -> anyhow::Result<BatchTrainType> {
        let inp = input.to_tensor(&self.device);
        let exp = expected.to_tensor(&self.device);
        self.train(inp, exp)
    }

    // pub fn train_full(&mut self, input: impl ToTensor<B,2>, expected: impl ToTensor<B,1>) -> anyhow::Result<TrainType> {
    //     let inp = input.to_tensor(&self.device);
    //     let exp = expected.to_tensor(&self.device);
    //     self.train_1t(inp.clone(), exp.clone())
    // }

    // pub fn train_1(&mut self, input: impl ToTensor<B,2>, expected: impl ToTensor<B,1>) -> anyhow::Result<TrainType> {
    //     self.train_1t(input.to_tensor(&self.device), expected.to_tensor(&self.device))
    // }

    // pub fn train_1t(&mut self, input: Tensor<B,2>, expected: Tensor<B,1>) -> anyhow::Result<TrainType> {
    //     self.train(input.unsqueeze(), expected.unsqueeze())
    // }

    // pub fn train_1(&mut self, input: [[f32; NUM_FEATURES]; SERIES_LENGTH], expected: [f32; MODEL_OUTPUT_WIDTH]) -> anyhow::Result<()> {
    //     let x = Tensor::from_floats(input, &self.device).unsqueeze();
    //     let y = Tensor::from_floats(expected, &self.device).unsqueeze();
    //     self.train_batch(x, y)
    // }

    // pub fn train_batch(&mut self, input: Tensor<B,3>, expected: Tensor<B,2>) -> anyhow::Result<()> {
    //     // let model = std::mem::replace(&mut self.model, None)
    //     //     .with_context(|| "No model in trainer")?;
    //     Ok(())
    // }

    fn train(&mut self, input: Tensor<B,3>, expected: Tensor<B,2>) -> anyhow::Result<BatchTrainType> {
        let model = self.model.take().with_context(|| "No model in trainer")?;
        let last_exp = last_index(&expected);

        // println!("  Expect: {:?}", &expected.to_data());
        let output = model.forward(input.clone());
        // println!("  Output: {:?}", &output.to_data());
        let loss = self.loss.forward(output, expected.clone(), Reduction::Mean);
        // let loss_simple: TrainType = loss.to_data().value[0].elem();
        let grads = GradientsParams::from_grads(loss.backward(), &model);
        let model2 = self.optimizer.step(self.config.learning_rate, model, grads);

        let output2 = model2.forward(input);
        // let last_out2 = last_index(&output2);
        // println!("Output: {}", last_out2);
        // println!("Expected: {}", last_exp);

        let mut losses = Vec::new();
        let mut builder = Builder::default();
        for (i, (out, exp)) in zip(output2.clone().iter_dim(0), expected.clone().iter_dim(0)).enumerate() {
            let loss: ModelFloat = self.loss.forward(out.clone(), exp.clone(), Reduction::Sum).to_data().value[0].elem();
            losses.push(loss);

            let mut expected_row = exp.squeeze::<1>(0).to_data().value.iter().map(|item| to_result_str(item.elem::<f32>())).collect::<Vec<_>>();
            expected_row.insert(0, String::default());
            expected_row.insert(0, format!("{i}: expected"));
            builder.push_record(expected_row);

            let mut output_row = out.squeeze::<1>(0).to_data().value.iter().map(|item| to_result_str(item.elem::<f32>())).collect::<Vec<_>>();
            output_row.insert(0, loss.to_string());
            output_row.insert(0, format!("{i}: output"));
            builder.push_record(output_row);
        }

        let mut columns = vec!["type".to_string(), "loss".to_string()];
        columns.extend((0..MODEL_OUTPUT_WIDTH).map(|i| i.to_string()));
        builder.insert_record(0, columns);
        let mut table = builder.build();
        table.with(Style::rounded());
        println!("{table}");

        // let loss2 = self.loss.forward(output2, expected, Reduction::Mean);
        // let loss2_simple: BatchTrainType = loss.to_data().value.iter().map(|item| item.elem()).collect(); //[0].elem();
        // println!("  Loss: {} -> {}", loss.to_data(), loss2.to_data());

        self.model = Some(model2);
        Ok(losses)
    }
}

fn last_index<B:Backend>(tensor: &Tensor<B,2>) -> Tensor<B,1> {
    let expected_shape = tensor.shape();
    let last_index = expected_shape.dims[0] - 1;
    tensor.clone().slice([last_index..(last_index+1), 0..MODEL_OUTPUT_WIDTH]).squeeze(0)
}

const STRING_ZERO: &str = "0";
const STRING_ONE: &str = "1";

fn to_result_str(x: f32) -> String {
    (if x >= 0.5 { STRING_ONE } else { STRING_ZERO }).to_string()
}

pub trait ToTensor<B: Backend, const N: usize> {
    fn to_tensor(self, device: &B::Device) -> Tensor<B,N>;
}

// impl<B: Backend, const N: usize, K: Into<Data<f32,N>>> ToTensor<B,N> for K {
//     fn to_tensor(self, device: &B::Device) -> Tensor<B,N> {
//         Tensor::from_floats(self, device)
//     }
// }

impl<B: Backend> ToTensor<B,3> for Vec<ModelInput> {
    fn to_tensor(self, device: &B::Device) -> Tensor<B,3> {
        // let x = Tensor::from_floats(self[0], device);
        let x = self.into_iter().map(|input| Tensor::from_floats(input, device)).collect();
        Tensor::stack(x, 0)
        // TryInto::<[ModelInput; ]
        // Tensor::from_floats(self, device)
    }
}

impl<B: Backend> ToTensor<B,2> for Vec<LabelType> {
    fn to_tensor(self, device: &B::Device) -> Tensor<B,2> {
        let x = self.into_iter().map(|input| Tensor::from_floats(input, device)).collect();
        Tensor::stack(x, 0)
    }
}
