use std::sync::{Arc, Mutex};

use anyhow::{bail, Context};
use burn::nn::loss::MseLoss;
use burn::optim::adaptor::OptimizerAdaptor;
use burn::optim::{AdamW, AdamWConfig, GradientsParams, Optimizer};
use burn::tensor::activation::log_softmax;
use burn::tensor::backend::AutodiffBackend;
use burn::prelude::*;

use itertools::Itertools;
use nn::loss::{BinaryCrossEntropyLoss, CrossEntropyLoss};
use shared_burn::model_persist::{load_config, save_config};
use shared_burn::{IsTensor, Model};
use shared_burn::model::TheModelConfig;

use shared_burn::model_persist::save_model;
use shared_types::ModelFloat;

#[derive(Config)]
pub struct TheTrainingConfig {
    #[config(default = 2)]
    pub num_epochs: usize,
    #[config(default = 64)]
    pub batch_size: usize,
    #[config(default = 12)]
    pub num_workers: usize,
    #[config(default = 42)]
    pub seed: u64,
    #[config(default = 1e-3)]
    pub learning_rate: f64,
    pub model_config: TheModelConfig,
    pub optimizer: AdamWConfig,
}

/// Regarding the Option model:
/// https://play.rust-lang.org/?version=stable&mode=debug&edition=2021&gist=2cfe1e7a3eaec1d6cf9ac41e9d4778fc
/// https://stackoverflow.com/questions/54293155/how-to-replace-the-value-of-a-struct-field-without-taking-ownership/54293313#54293313
pub struct Trainer<B: AutodiffBackend, M: Model + burn::module::AutodiffModule<B>, L: LossCalc<B,M>> {
    train_config: TheTrainingConfig,
    pub model: Arc<Mutex<Option<M>>>,
    optimizer: OptimizerAdaptor<AdamW<B::InnerBackend>, M, B>,
    loss_calc: L,
    // MseLoss<B>
    // validation_loss: MseLoss<<B as AutodiffBackend>::InnerBackend>,
}

// pub fn save_configs() -> anyhow::Result<()> {
//     let model_config = Self::new_config();
//     save_model_config(&model_config)?;
//     let train_config = TheTrainingConfig::new(model_config, AdamWConfig::new());
//     save_train_config(&train_config)?;
//     Ok(())
// }

impl<B: AutodiffBackend, M: Model + burn::module::AutodiffModule<B>, L: LossCalc<B,M>> Trainer<B,M,L> {
    pub fn new(train_config: TheTrainingConfig, model: M, loss: L) -> anyhow::Result<Self> {
        // let loss = MseLoss::new();
        // let validation_loss = MseLoss::new();
        let optimizer = AdamWConfig::new().init::<B,M>();

        Ok(Self { train_config, model: Arc::new(Mutex::new(Some(model))), loss_calc: loss, optimizer })
    }

    /// # Shapes
    ///   - Input [batch_size, SERIES1_LENGTH, NUM_FEATURES]
    ///   - Expected [batch_size, MODEL_OUTPUT_WIDTH]
    ///   - Output [batch_size] of losses
    // pub fn train_batch(&mut self, input: impl ToTensor<B,3>, expected: impl ToTensor<B,2>, display_result: bool)
    //         -> anyhow::Result<(Vec<f32>,Vec<ModelOutput>)> {
    //     let inp = input.to_tensor(&self.device);
    //     let exp = expected.to_tensor(&self.device);
    //     self.train(inp, exp, display_result).map(|(losses, outputs)| {
    //         (tensor_to_vec_f32(losses), tensor_to_output(outputs))
    //     })
    // }

    pub fn train(&mut self, input: M::Input, expected: M::Output) -> anyhow::Result<(M::Output, Tensor<B,1>, ModelFloat)> {
// anyhow::Result<(Tensor<B::InnerBackend,1>,Tensor<B::InnerBackend,2>)> {
        let keep_expected = expected.clone().noop();

        let mut model_guard = match self.model.lock() {
            Ok(model) => model,
            Err(_) => bail!("Could not get lock on model"),
        };

        let model = model_guard.take().ok_or(anyhow::anyhow!("No model in trainer"))?;

        // let last_exp = last_index(&expected);

        let output = model.forward(input);
        let keep_output = output.clone();

        let losses = self.loss_calc.for_each_in_batch(output, expected);
        let keep_losses = losses.clone();
        let loss = losses.mean();
        let grads = GradientsParams::from_grads(loss.backward(), &model);
        let loss_value = loss.into_data().convert::<f32>().value[0];
        if !loss_value.is_finite() || loss_value > 1000.0 {
            let out = keep_output.noop().clone().into_data().convert::<f32>().value;
            println!("  expected: {}", keep_expected.into_data());
            bail!("Loss is NaN or too large: {}, for outputs (minmax: {:?}): {:?}", loss_value, out.iter().minmax(), out);
        }
        let model2 = self.optimizer.step(self.train_config.learning_rate, model, grads);

        // Not running validation will make the loss and output one iteration behind, but it will speed things up.
        // let validation_model = model2.valid();
        // let validation_input = input.valid();
        // let validation_output = validation_model.forward(validation_input);
        // let validation_expected = expected.valid();
        // let validation_losses = self.validation_loss.forward_no_reduction(validation_output.clone(), validation_expected.clone()).mean_dim(1).squeeze(1);

        // if display_result {
            // print_compare_table(validation_output.clone(), validation_expected, &validation_losses);
        // }

        // self.model = Some(model2);
        *model_guard = Some(model2);
        // Ok((validation_losses, validation_output))
        Ok((keep_output, keep_losses, loss_value))
    }

    // pub fn save_model(&self) -> anyhow::Result<()> {
    //     save_model(self.model.as_ref().unwrap())?;
    //     Ok(())
    // }
}

// pub fn tensor_to_output<B: Backend>(tensor: Tensor<B, 2>) -> Vec<ModelOutput> {
//     let flat = tensor.into_data().convert::<f32>().value;
//     flat.chunks_exact(MODEL_OUTPUT_WIDTH)
//         .map(TryInto::try_into)
//         .map(Result::unwrap)
//         .collect()
// }

// // fn last_index<B:Backend>(tensor: &Tensor<B,2>) -> Tensor<B,1> {
// //     let expected_shape = tensor.shape();
// //     let last_index = expected_shape.dims[0] - 1;
// //     tensor.clone().slice([last_index..(last_index+1), 0..MODEL_OUTPUT_WIDTH]).squeeze(0)
// // }

// pub trait ToTensor<B: Backend, const N: usize> {
//     fn to_tensor(self, device: &B::Device) -> Tensor<B,N>;
// }

// // impl<B: Backend, const N: usize, K: Into<Data<f32,N>>> ToTensor<B,N> for K {
// //     fn to_tensor(self, device: &B::Device) -> Tensor<B,N> {
// //         Tensor::from_floats(self, device)
// //     }
// // }

// impl<B: Backend> ToTensor<B,3> for Vec<ModelInput> {
//     fn to_tensor(self, device: &B::Device) -> Tensor<B,3> {
//         let x = self.into_iter().map(|input| Tensor::from_floats(input, device)).collect();
//         Tensor::stack(x, 0)
//     }
// }

// impl<B: Backend> ToTensor<B,2> for ModelInput {
//     fn to_tensor(self, device: &B::Device) -> Tensor<B,2> {
//         Tensor::from_floats(self, device)
//     }
// }

// impl<B: Backend> ToTensor<B,2> for Vec<LabelType> {
//     fn to_tensor(self, device: &B::Device) -> Tensor<B,2> {
//         let x = self.into_iter().map(|input| Tensor::from_floats(input, device)).collect();
//         Tensor::stack(x, 0)
//     }
// }

// impl<B: Backend> ToTensor<B,1> for LabelType {
//     fn to_tensor(self, device: &B::Device) -> Tensor<B,1> {
//         Tensor::from_floats(self, device)
//         // let x = self.into_iter().map(|input| Tensor::from_floats(input, device)).collect();
//         // Tensor::stack(x, 0)
//     }
// }


const CONFIG_TRAIN: &str = "config-train.json";
pub fn save_train_config(config: &TheTrainingConfig) -> anyhow::Result<()> {
    save_config("training", CONFIG_TRAIN, config)
}

pub fn load_train_config() -> anyhow::Result<TheTrainingConfig> {
    load_config("training", CONFIG_TRAIN)
}

pub trait LossCalc<B:Backend, M: Model> {
    fn for_each_in_batch(&self, output: M::Output, expected: M::Output) -> Tensor<B,1>;
}

// pub struct MseLossCalc<B:Backend>(pub MseLoss<B>);

// impl<B:Backend,M:Model> LossCalc<B,M> for MseLossCalc<B>
// where M::Output: IsTensor<B,2>
// {
//     fn for_each_in_batch(&self, output: M::Output, expected: M::Output) -> Tensor<B,1> {
//         self.0.forward_no_reduction(output.noop(), expected.noop()).mean_dim(1).squeeze(1)
//     }
// }

pub struct BinaryCrossEntropyLossCalc<B:Backend>(pub BinaryCrossEntropyLoss<B>);

impl<B:Backend,M:Model> LossCalc<B,M> for BinaryCrossEntropyLossCalc<B>
where M::Output: IsTensor<B,2>
{
    fn for_each_in_batch(&self, output: M::Output, expected: M::Output) -> Tensor<B,1> {
        // TODO: very inefficient to do this conversion to int here
        let output = output.noop();
        let expected = expected.noop();
        let batch_size = expected.dims()[0];
        let s1: f32 = expected.clone().sum().into_scalar().elem();
        let exp = expected.clone().int();
        let s2: i32 = exp.clone().sum().into_scalar().elem();
        if !(s1 >= 0.0 && s2 >= 0 && (s1 as i32) == s2) {
            println!("Conversion to int tensor resulted in non-matching sums: f32: {}, i32: {}", s1, s2);
            panic!("stop");
        }
        if s1 == 0.0 || s2 == 0 {
            println!("Warning, expected tensor sum was 0: f32: {}, i32: {}", s1, s2);
        }
        // println!("LOSS COMPARE:");
        // println!("  expected: {:?}", expected.clone().into_data());
        // println!("  exp: {:?}", exp.clone().into_data());
        // println!("  output: {:?}", output.clone().into_data());
        let res = self.0.forward_no_reduction(output, exp);
        assert!(res.dims() == [batch_size]);
        res
        // // self.0.forward(logits, targets)
        // let expected = expected.noop();
        // let output = output.noop();
        // let [batch_size, _] = expected.dims();

        // let tensor = log_softmax(output, 1);
        // let tensor = tensor.gather(1, targets.clone().reshape([batch_size, 1]));

        // let tensor = Self::apply_mask_1d(tensor.reshape([batch_size]), mask);
        // tensor.mean().neg()
    }
}
