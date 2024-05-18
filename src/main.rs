use series_store::*;
use shared_types::*;


fn main() -> anyhow::Result<()> {
    let logger = StdoutLogger::boxed();

    let mut reader = SeriesReader::new(logger)?;
    let topic = Topic::new("raw", "SPY", "quote");
    reader.subscribe(&topic, Offset::Beginning)?;
    let msgs = reader.read(100)?;
    println!("count: {}", msgs.len());
    println!("first: {:?}", String::from_utf8(msgs[0].payload().unwrap().into())?);
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
pub struct Quote {
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