use burn::{nn::{Linear, LinearConfig, Relu}, prelude::*};

#[derive(Config, Debug)]
pub(crate) struct TheModelConfig {
    num_features: usize,
    series_length: usize,
    num_classes: usize,
}

impl TheModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> TheModel<B> {
        // let output = LinearConfig::new(self.num_features * self.series_length, self.num_classes).init(device);
        let hidden_width = 1024;
        let input = LinearConfig::new(self.num_features * self.series_length, hidden_width).init(device);
        let layer1 = LinearConfig::new(hidden_width, hidden_width).init(device);
        let output = LinearConfig::new(hidden_width, self.num_classes).init(device);
        TheModel { input, layer1, output, act: Relu::new() }
    }
}

#[derive(Module, Debug)]
pub(crate) struct TheModel<B: Backend> {
    input: Linear<B>,
    layer1: Linear<B>,
    output: Linear<B>,
    act: Relu
}

/// # Shapes
///   - Input [batch_size, SERIES_LENGTH, NUM_FEATURES]
///   - Output [batch_size, MODEL_OUTPUT_WIDTH]
impl<B: Backend> TheModel<B> {
    pub fn forward(&self, input: Tensor<B,3>) -> Tensor<B,2> {
        // let a = self.output.forward(input.flatten(1, 2));
        // self.act.forward(a)
        let a = self.input.forward(input.flatten(1, 2));
        let b = self.layer1.forward(a);
        let c = self.output.forward(b);
        // burn::tensor::activation::softmax(c, 0)
        // burn::tensor::activation::mish(c).clamp(0f32, 1f32)
        burn::tensor::activation::mish(c)
        // self.act.forward(c)
    }
}