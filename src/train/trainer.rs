use burn::nn::loss::MseLoss;
use burn::optim::adaptor::OptimizerAdaptor;
use burn::optim::{AdamW, AdamWConfig, GradientsParams, Optimizer};
use burn::tensor::backend::AutodiffBackend;
use burn::prelude::*;

use shared_burn::model_persist::{load_config, save_config};
use shared_burn::{IsTensor, Model};
use shared_burn::model::TheModelConfig;

use shared_burn::model_persist::save_model;

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
    #[config(default = 1e-4)]
    pub learning_rate: f64,
    pub model_config: TheModelConfig,
    pub optimizer: AdamWConfig,
}

/// Regarding the Option model:
/// https://play.rust-lang.org/?version=stable&mode=debug&edition=2021&gist=2cfe1e7a3eaec1d6cf9ac41e9d4778fc
/// https://stackoverflow.com/questions/54293155/how-to-replace-the-value-of-a-struct-field-without-taking-ownership/54293313#54293313
pub struct Trainer<B: AutodiffBackend, M: Model + burn::module::AutodiffModule<B>, L: LossCalc<B,M>> {
    train_config: TheTrainingConfig,
    model: Option<M>,
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

        Ok(Self { train_config, model: Some(model), loss_calc: loss, optimizer })
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

    pub fn train(&mut self, input: M::Input, expected: M::Output) -> anyhow::Result<(Tensor<B,1>, M::Output)> {
// anyhow::Result<(Tensor<B::InnerBackend,1>,Tensor<B::InnerBackend,2>)> {
        let model = self.model.take().ok_or(anyhow::anyhow!("No model in trainer"))?;
        // let last_exp = last_index(&expected);

        // println!("  Expect: {:?}", &expected.to_data());
        let output = model.forward(input);
        let keep_output = output.clone();
        // println!("  Output: {:?}", &output.to_data());

        let losses = self.loss_calc.for_each_in_batch(output, expected);
        let keep_losses = losses.clone();
        let loss = losses.mean();
        // let loss = self.loss_calc.forward(output, expected.clone(), Reduction::Mean);
        let grads = GradientsParams::from_grads(loss.backward(), &model);
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

        self.model = Some(model2);
        // Ok((validation_losses, validation_output))
        Ok((keep_losses, keep_output))
    }

    pub fn save_model(&self) -> anyhow::Result<()> {
        save_model(self.model.as_ref().unwrap())?;
        Ok(())
    }
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

pub struct MseLossCalc<B:Backend>(pub MseLoss<B>);

impl<B:Backend,M:Model> LossCalc<B,M> for MseLossCalc<B>
where M::Output: IsTensor<B,2>
{
    // fn forward_no_reduction(&self, output: M::Output, expected: M::Output) {
    fn for_each_in_batch(&self, output: M::Output, expected: M::Output) -> Tensor<B,1> {
        self.0.forward_no_reduction(output.noop(), expected.noop()).mean_dim(1).squeeze(1)
    }
}
