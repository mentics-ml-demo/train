use std::env;

use shared_types::data_info::CURRENT_VERSION;
use train::data::*;

#[tokio::main(flavor = "multi_thread", worker_threads = 10)]
async fn main() -> anyhow::Result<()> {
    let mut count = 1000;
    let mut reset = false;
    let args: Vec<String> = env::args().collect();
    if args.len() > 1 {
        let mut index = 1;
        let mut arg = &args[index];
        if arg == "reset" {
            reset = true;
            index += 1;
        }
        arg = &args[index];
        if let Ok(arg_count) = arg.parse::<usize>() {
            count = arg_count;
        }
    }

    // Trainer::<TheAutodiffBackend>::save_configs()?;
    let mut data = make_mgr(CURRENT_VERSION, reset).await?;
    data.run(count).await?;

    Ok(())
}
