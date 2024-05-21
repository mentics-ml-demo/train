pub mod data;
pub mod trainer;

use shared_types::*;
use series_store::*;
use kv_store::*;
use data::*;
use trainer::*;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let logger = StdoutLogger::boxed();
    let topic = Topic::new("raw", "SPY", "quote");
    let series = SeriesReader::new(logger)?;
    let store = KVStore::new().await?;

    let trainer = Trainer::new();
    let data = DataMgr::new(CURRENT_VERSION, series, store, topic, trainer).await?;
    data.run().await?;

    Ok(())
}
