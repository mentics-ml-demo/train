pub mod data;
pub mod train;

use shared_types::*;
use series_store::*;
use kv_store::*;
use data::*;
use train::trainer::*;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let logger = StdoutLogger::boxed();
    let topic = Topic::new("raw", "SPY", "quote");
    let series = SeriesReader::new(logger)?;
    let store = KVStore::new().await?;

    let trainer = make_trainer();
    let mut data = DataMgr::new(CURRENT_VERSION, series, store, topic, trainer).await?;
    data.run().await?;

    Ok(())
}
