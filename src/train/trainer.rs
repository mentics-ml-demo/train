use anyhow::Context;
use burn::module::AutodiffModule;
use burn::nn::loss::{MseLoss, Reduction};
use burn::optim::adaptor::OptimizerAdaptor;
use burn::optim::{AdamW, AdamWConfig, GradientsParams, Optimizer};
use burn::tensor::backend::AutodiffBackend;
use burn::prelude::*;

use shared_types::*;
use crate::model_persist::*;
use crate::output::print_compare_table;
use crate::train::model::*;
use crate::TheAutodiffBackend;

#[derive(Config)]
pub(crate) struct TheTrainingConfig {
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
    pub model_config: TheModelConfig,
    pub optimizer: AdamWConfig,
}

/// Regarding the Option model:
/// https://play.rust-lang.org/?version=stable&mode=debug&edition=2021&gist=2cfe1e7a3eaec1d6cf9ac41e9d4778fc
/// https://stackoverflow.com/questions/54293155/how-to-replace-the-value-of-a-struct-field-without-taking-ownership/54293313#54293313
pub struct Trainer<B: AutodiffBackend> {
    device: B::Device,
    train_config: TheTrainingConfig,
    model: Option<TheModel<B>>,
    optimizer: OptimizerAdaptor<AdamW<B::InnerBackend>, TheModel<B>, B>,
    loss_calc: MseLoss<B>,
    validation_loss: MseLoss<<B as AutodiffBackend>::InnerBackend>,
}

pub fn make_trainer(new_model: bool) -> anyhow::Result<Trainer<TheAutodiffBackend>> {
    let device = burn::backend::wgpu::WgpuDevice::default();
    Trainer::new(device, new_model)
}

impl<B: AutodiffBackend> Trainer<B> {
    pub fn save_configs() -> anyhow::Result<()> {
        let model_config = TheModelConfig::new(NUM_FEATURES, SERIES_SIZE, MODEL_OUTPUT_WIDTH);
        save_model_config(&model_config)?;
        let train_config = TheTrainingConfig::new(model_config, AdamWConfig::new());
        save_train_config(&train_config)?;
        Ok(())
    }

    pub fn new(device: B::Device, new_model: bool) -> anyhow::Result<Self> {
        let (train_config, model) = if new_model {
            let model_config = TheModelConfig::new(NUM_FEATURES, SERIES_SIZE, MODEL_OUTPUT_WIDTH);
            save_model_config(&model_config)?;
            let model = model_config.init::<B>(&device);
            save_model(&model)?;
            let train_config = TheTrainingConfig::new(model_config, AdamWConfig::new());
            save_train_config(&train_config)?;
            (train_config, model)
        } else {
            (load_train_config()?, load_model(&device)?)
            // let model_path = artifacts_dir()?.join("model.mpk");
            // if model_path.try_exists()? {
            //     println!("Loading model... (from {:?})", &model_path);
            //     let record = CompactRecorder::new().load(model_path, &device).with_context(|| "Error loading model")?;
            //     model = config.model.init::<B>(&device).load_record(record);
            //     println!("  model loaded.")
            // }
        };

        let loss = MseLoss::new();
        let validation_loss = MseLoss::new();
        let optimizer = AdamWConfig::new().init::<B,TheModel<B>>();

        Ok(Self { device, train_config, model: Some(model), loss_calc: loss, validation_loss, optimizer })
    }

    pub fn save_model(&self) -> anyhow::Result<()> {
        save_model(self.model.as_ref().unwrap())
    }

    /// # Shapes
    ///   - Input [batch_size, SERIES_LENGTH, NUM_FEATURES]
    ///   - Expected [batch_size, MODEL_OUTPUT_WIDTH]
    ///   - Output [batch_size] of losses
    pub fn train_batch(&mut self, input: impl ToTensor<B,3>, expected: impl ToTensor<B,2>, display_result: bool) -> anyhow::Result<BatchTrainResultType> {
        let inp = input.to_tensor(&self.device);
        let exp = expected.to_tensor(&self.device);
        self.train(inp, exp, display_result).map(tensor_to_vec_f32)
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

    fn train(&mut self, input: Tensor<B,3>, expected: Tensor<B,2>, display_result: bool) ->
    // anyhow::Result<Tensor<B,1>>
    anyhow::Result<Tensor<B::InnerBackend,1>>
    {
        let model = self.model.take().with_context(|| "No model in trainer")?;
        // let last_exp = last_index(&expected);

        // println!("  Expect: {:?}", &expected.to_data());
        let output = model.forward(input.clone());
        // println!("  Output: {:?}", &output.to_data());
        let loss = self.loss_calc.forward(output, expected.clone(), Reduction::Mean);
        // let loss_simple: TrainType = loss.to_data().value[0].elem();
        let grads = GradientsParams::from_grads(loss.backward(), &model);
        let model2 = self.optimizer.step(self.train_config.learning_rate, model, grads);

        let validation_model = model2.valid();
        let validation_input = input.valid();
        let validation_output = validation_model.forward(validation_input);
        let validation_expected = expected.valid();
        let validation_losses = self.validation_loss.forward_no_reduction(validation_output.clone(), validation_expected.clone()).mean_dim(1).squeeze(1);

        if display_result {
            print_compare_table(validation_output, validation_expected, &validation_losses);
        }


        self.model = Some(model2);
        Ok(validation_losses)
    }
}

fn tensor_to_vec_f32<B: Backend>(tensor: Tensor<B, 1>) -> Vec<f32> {
    tensor.into_data().convert::<f32>().value
}

// fn last_index<B:Backend>(tensor: &Tensor<B,2>) -> Tensor<B,1> {
//     let expected_shape = tensor.shape();
//     let last_index = expected_shape.dims[0] - 1;
//     tensor.clone().slice([last_index..(last_index+1), 0..MODEL_OUTPUT_WIDTH]).squeeze(0)
// }

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
        let x = self.into_iter().map(|input| Tensor::from_floats(input, device)).collect();
        Tensor::stack(x, 0)
    }
}

impl<B: Backend> ToTensor<B,2> for ModelInput {
    fn to_tensor(self, device: &B::Device) -> Tensor<B,2> {
        Tensor::from_floats(self, device)
    }
}

impl<B: Backend> ToTensor<B,2> for Vec<LabelType> {
    fn to_tensor(self, device: &B::Device) -> Tensor<B,2> {
        let x = self.into_iter().map(|input| Tensor::from_floats(input, device)).collect();
        Tensor::stack(x, 0)
    }
}

impl<B: Backend> ToTensor<B,1> for LabelType {
    fn to_tensor(self, device: &B::Device) -> Tensor<B,1> {
        Tensor::from_floats(self, device)
        // let x = self.into_iter().map(|input| Tensor::from_floats(input, device)).collect();
        // Tensor::stack(x, 0)
    }
}
