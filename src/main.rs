pub mod data;
pub mod train;

use std::env;

use anyhow::Context;
use data::*;
use shared_types::CURRENT_VERSION;
use train::trainer::*;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let h = home::home_dir().with_context(|| "Could not get user home directory")?;
    let path = h.join("data").join("models").join("oml");

    let args: Vec<String> = env::args().collect();
    println!("Found args: {:?}", args);
    if args.len() > 1 {
        let arg = &args[1];
        if arg == "reset" {
            println!("Deleting artifacts: {:?}", &path);
            std::fs::remove_dir_all(&path)?;
        }
    }

    let mut data = make_mgr(CURRENT_VERSION, &path).await?;
    data.run().await?;

    Ok(())
}
