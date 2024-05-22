pub mod data;
pub mod train;

use std::env;

use anyhow::Context;
use shared_types::*;
use series_store::*;
use kv_store::*;
use data::*;
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

    let logger = StdoutLogger::boxed();
    let topic = Topic::new("raw", "SPY", "quote");
    let series = SeriesReader::new(logger)?;
    let store = KVStore::new().await?;

    let trainer = make_trainer(path)?;
    let mut data = DataMgr::new(CURRENT_VERSION, series, store, topic, trainer).await?;
    data.run().await?;

    Ok(())
}
