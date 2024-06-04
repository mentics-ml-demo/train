use std::iter::zip;
use tabled::{builder::Builder, settings::Style};
use burn::tensor::{backend::Backend, ElementConversion, Tensor};

use shared_types::MODEL_OUTPUT_WIDTH;

pub fn print_compare_table<B:Backend>(
        output: Tensor<B,2>,
        expected: Tensor<B,2>,
        losses: &Tensor<B,1> //&[f32]
) {
    let loss_vec = losses.to_data().convert::<f32>().value;
    let mut builder = Builder::default();
    for (i, (out, exp)) in zip(output.clone().iter_dim(0), expected.clone().iter_dim(0)).enumerate() {
        // let loss: ModelFloat = self.loss.forward(out.clone(), exp.clone(), Reduction::Sum).to_data().value[0].elem();
        // losses.push(loss);
        let loss = loss_vec[i];

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
}

const STRING_ZERO: &str = "0";
const STRING_ONE: &str = "1";

fn to_result_str(x: f32) -> String {
    (if x >= 0.5 { STRING_ONE } else { STRING_ZERO }).to_string()
}
