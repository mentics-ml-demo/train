use anyhow::anyhow;
use burn::{nn::loss::{MseLoss, Reduction}, tensor::{backend::Backend, Tensor}};

use shared_types::*;
use crate::{model_persist::load_model, output::print_compare_table, train::{model::TheModel, trainer::ToTensor}, TheBackend};

pub fn make_inferer() -> anyhow::Result<Inferer<TheBackend>> {
    let device = burn::backend::wgpu::WgpuDevice::default();
    Inferer::new(device)
}

pub struct Inferer<B:Backend> {
    device: B::Device,
    model: TheModel<B>,
    loss_calc: MseLoss<B>,
}

impl<B:Backend> Inferer<B> {
    pub fn new(device: B::Device) -> anyhow::Result<Self> {
        let model = load_model(&device)?;
        let loss = MseLoss::new();

        println!("Is autodiff enabled for infer backend: {}", B::ad_enabled());
        assert!(!B::ad_enabled());

        Ok(Self { device, model, loss_calc: loss })
    }

    pub fn infer_and_check(&self, input: impl ToTensor<B,2>, expected: impl ToTensor<B,1>, display_result: bool) -> TrainResultType {
        let input_tensor = input.to_tensor(&self.device).unsqueeze();
        let output = self.infer_batch(input_tensor).squeeze(0);
        let expected = expected.to_tensor(&self.device);
        let losses = self.loss_calc.forward(output.clone(), expected.clone(), Reduction::Mean);

        if display_result {
            print_compare_table(output.unsqueeze(), expected.unsqueeze(), &losses);
        }

        let v = losses.into_data().convert::<f32>().value;
        assert!(v.len() == 1);

        v[0]
    }

    pub fn infer_1(&self, input: impl ToTensor<B,2>) -> anyhow::Result<ModelOutput> {
        let input_tensor = input.to_tensor(&self.device).unsqueeze();
        let output = self.infer_batch(input_tensor);
        output.into_data().convert::<f32>().value.try_into().map_err(|e| anyhow!("Error converting inference output to ModelOutput type {:?}", e))
    }

    pub fn infer_batch(&self, input: Tensor<B,3>) -> Tensor<B,2> {
        self.model.forward(input)
    }
}