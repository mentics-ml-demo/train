use shared_types::*;

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

pub struct Trainer {

}

impl Trainer {
    pub fn new() -> Self {
        Self { }
    }

    pub fn train(&self, quotes: Vec<QuoteEvent>, label: Label) {
        println!("{:?}", label);
    }
}