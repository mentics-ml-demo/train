use burn::{nn::{Linear, LinearConfig}, prelude::*};

#[derive(Config, Debug)]
pub(crate) struct TheModelConfig {
    num_features: usize,
    series_length: usize,
    num_classes: usize,
}

impl TheModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> TheModel<B> {
        let output = LinearConfig::new(self.num_features * self.series_length, self.num_classes).init(device);
        TheModel { output }
    }
}

#[derive(Module, Debug)]
pub(crate) struct TheModel<B: Backend> {
    output: Linear<B>,
}

/// # Shapes
///   - Input [batch_size, SERIES_LENGTH, NUM_FEATURES]
///   - Output [batch_size, MODEL_OUTPUT_WIDTH]
impl<B: Backend> TheModel<B> {
    pub fn forward(&self, input: Tensor<B,3>) -> Tensor<B,2> {
        self.output.forward(input.flatten(1, 2))
    }
}