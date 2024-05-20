pub mod trainer;

use shared_types::*;
use series_store::*;
use kv_store::*;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let logger = StdoutLogger::boxed();
    let topic = Topic::new("raw", "SPY", "quote");

    let mut series = SeriesReader::new(logger)?;
    series.subscribe(&topic, Offset::Beginning)?;

    let store = KVStore::new().await?;

    let msgs = series.read_count(SERIES_LENGTH)?;
    println!("count: {}", msgs.len());
    println!("first: {:?}", msg_to::<QuoteEvent>(msgs.first().unwrap())?);
    Ok(())
}

///
/*
{
"type":"quote"
"symbol":"SPX"
"bid":5249.61
"bidsz":0
"bidexch":""
"biddate":"1715716641000"
"ask":5250.74
"asksz":0
"askexch":""
"askdate":"1715716641000"
}
*/
///
pub struct InputQuote {
    pub bid: f32,
    pub bid_size: f32,
    pub bid_ts: u64,
    pub ask: f32,
    pub ask_size: f32,
    pub ask_ts: u64,
}

pub struct Trade {

}

pub struct TimeSale {

}