use burn::{nn::{Linear, LinearConfig, Relu}, prelude::*, tensor::activation::mish};
use nn::{Dropout, DropoutConfig};

#[derive(Config, Debug)]
pub(crate) struct TheModelConfig {
    num_features: usize,
    series_length: usize,
    num_classes: usize,
}

impl TheModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> TheModel<B> {
        let dropout_factor = 0.2;
        // let output = LinearConfig::new(self.num_features * self.series_length, self.num_classes).init(device);
        let hidden_width = 1024;
        let input = LinearConfig::new(self.num_features * self.series_length, hidden_width).init(device);
        let layer1 = LinearConfig::new(hidden_width, hidden_width).init(device);
        let dropout = DropoutConfig::new(dropout_factor).init();
        let layer2 = LinearConfig::new(hidden_width, hidden_width).init(device);
        let output = LinearConfig::new(hidden_width, self.num_classes).init(device);
        TheModel { input, layer1, dropout, layer2, output, act: Relu::new() }
    }
}

#[derive(Module, Debug)]
pub(crate) struct TheModel<B: Backend> {
    input: Linear<B>,
    layer1: Linear<B>,
    dropout: Dropout,
    layer2: Linear<B>,
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



        let mut x = self.input.forward(input.flatten(1, 2));
        x = mish(x);
        x = self.layer1.forward(x);
        x = mish(x);
        x = self.dropout.forward(x);
        x = self.layer2.forward(x);
        x = mish(x);
        x = self.output.forward(x);
        x = mish(x);
        x


        // burn::tensor::activation::softmax(c, 0)
        // burn::tensor::activation::mish(c).clamp(0f32, 1f32)
        // burn::tensor::activation::mish(c)
        // self.act.forward(c)
    }
}

// fn run_in_order<B:Backend,const D: usize,T:Default>(funcs: Vec<impl FnOnce(T) -> T>) -> T {
//     let mut x = Default::default();
//     for f in funcs {
//         x = f(x);
//     }
//     x
// }