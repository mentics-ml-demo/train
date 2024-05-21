use burn::{data::{dataloader::batcher::Batcher, dataset::Dataset}, prelude::*};

#[derive(Clone,Debug)]
pub struct TheInput {
    pub input: [[[f32; 4]; 3]; 2],
    pub actual: [f32; 2],
}

#[derive(Clone)]
pub struct TheBatcher<B: Backend> {
    device: B::Device,
}

impl<B: Backend> TheBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}

#[derive(Clone, Debug)]
pub struct TheBatch<B: Backend> {
    pub inputs: Tensor<B, 4>,
    pub actuals: Tensor<B, 2>,
}

pub struct TheDataset {
    data: Vec<TheInput>,
}

impl TheDataset {
    pub fn train() -> TheDataset {
        let item_input = [
            // gen_floats2::<3,4>(),
            // gen_floats2::<3,4>(),
            [[0.1f32, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 8.0], [9.0, 10.0, 11.0, 12.0]],
            [[101.0, 102.0, 103.0, 104.0], [105.0, 106.0, 107.0, 108.0], [109.0, 110.0, 111.0, 112.0]]
        ];
        let item_actual = [0.7f32, 0.13];

        let data: Vec<TheInput> = (0..8192).map(|_| TheInput { input: item_input, actual: item_actual }).collect();

        TheDataset { data }
    }

    pub fn test() -> TheDataset {
        let input = [
            // gen_floats2::<3,4>(),
            // gen_floats2::<3,4>(),
            [[1.0f32, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]],
            [[101.0, 102.0, 103.0, 104.0], [105.0, 106.0, 107.0, 108.0], [109.0, 110.0, 111.0, 112.0]]
        ];
        let actual = [0.7f32, 0.13];
        TheDataset { data: vec!(TheInput { input, actual }) }
    }
}

impl Dataset<TheInput> for TheDataset {
    fn get(&self, index: usize) -> Option<TheInput> {
        self.data.get(index).cloned()
    }

    fn len(&self) -> usize {
        self.data.len()
    }
}

impl<B: Backend> Batcher<TheInput, TheBatch<B>> for TheBatcher<B> {
    fn batch(&self, data: Vec<TheInput>) -> TheBatch<B> {
        let input_iter = data.iter().map(|input|
            Tensor::from_floats(input.input, &self.device));
        // let inputs = Tensor::cat(input_iter.collect(), 0);
        let inputs = Tensor::stack(input_iter.collect(), 0);

        let actual_iter = data.iter().map(|input|
            Tensor::from_floats(input.actual, &self.device));
        // let actuals = Tensor::cat(actual_iter.collect(), 0);
        let actuals = Tensor::stack(actual_iter.collect(), 0);

        TheBatch { inputs, actuals }
    }
}
