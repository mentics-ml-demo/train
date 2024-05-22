use anyhow::Context;
use shared_types::*;
use series_store::*;
use kv_store::*;

use crate::{train::trainer::Trainer, TheAutodiffBackend};

pub struct DataMgr {
    version: VersionType,
    series: SeriesReader,
    store: KVStore,
    data_topic: Topic,
    trainer: Trainer<TheAutodiffBackend>,
}

impl DataMgr {
    pub async fn new(version: VersionType, mut series: SeriesReader, store: KVStore, topic: Topic, trainer: Trainer<TheAutodiffBackend>) -> anyhow::Result<Self> {
        store.max_trained_event_id(version).await?;
        series.subscribe(&topic, Offset::Beginning)?;

        // let msgs = series.read_count(SERIES_LENGTH)?;
        // println!("count: {}", msgs.len());
        // println!("first: {:?}", msg_to::<QuoteEvent>(msgs.first().unwrap())?);
        Ok(Self { version, series, store, data_topic: topic, trainer })
    }

    pub async fn run(&mut self) -> anyhow::Result<u64> {
        let mut trained_event_id = self.store.max_trained_event_id(self.version).await?.unwrap_or(0);
        let mut count = 0;

        // let labeleds = self.store.labeled_next(self.version, trained_event_id).await?;
        // let labeled = labeleds.unwrap();
        // self.series.seek(&self.data_topic, labeled.partition, Offset::Offset(labeled.offset - SERIES_LENGTH as i64 + 1))?;
        // let quotes: Vec<QuoteEvent> = self.series.read_count_into(SERIES_LENGTH)?;
        // let arr = quotes_to_arrays(quotes);

        loop {
            let labeleds = self.store.labeled_next(self.version, trained_event_id).await?;
            if labeleds.is_none() {
                break;
            }

            let labeled = labeleds.unwrap();
            self.series.seek(&self.data_topic, labeled.partition, Offset::Offset(labeled.offset - SERIES_LENGTH as i64 + 1))?;
            println!("{}: read {SERIES_LENGTH} events at offset: {}", labeled.event_id, labeled.offset);
            let quotes: Vec<QuoteEvent> = self.series.read_count_into(SERIES_LENGTH)?;
            assert!(quotes.last().unwrap().event_id == labeled.event_id);
            let arr = quotes_to_arrays(quotes);
            self.trainer.train_full(arr, labeled.label.value)?;
            trained_event_id = labeled.event_id;

            count += 1;
            if count > 200 { break; }
        }
        self.trainer.save_model()?;
        Ok(count)
    }

    // fn convert_to_input(quotes: Vec<QuoteEvent>) {
    //     quotes.map()
    // }
}

fn quotes_to_arrays(v: Vec<QuoteEvent>) -> [[f32; NUM_FEATURES]; SERIES_LENGTH] {
    assert!(v.len() == SERIES_LENGTH);
    // let it = v.iter().map(|x| x.into());
    // let res: Array2<f32> = Array::from_iter(it);

    // unwrap is same as above assertion
    let base = v.last().unwrap();
    let mut arr = [[0f32; NUM_FEATURES]; SERIES_LENGTH];
    for i in 0..SERIES_LENGTH {
        let q = &v[i];
        arr[i][0] = adjust(base.bid / q.bid);
        arr[i][1] = adjust(base.bid / q.ask);
    }

    // let res = [[0f32; NUM_FEATURES]; SERIES_LENGTH];
    // for i in 0..SERIES_LENGTH {
    //     let q = v[i];
    //     res[i][0] = adjust(base.bid / q.bid);
    //     res[i][1] = adjust(base.ask / q.ask);
    // }

    arr
}

// fn label_to_arr(label: Label) -> Array1<f32> {
//     let mut arr = Array1::zeros(MODEL_OUTPUT_WIDTH);
//     arr[0] = adjust(label.value[0]);
//     arr[1] = adjust(label.value[1]);
//     arr
// }

fn adjust(x: f32) -> f32 {
    x.clamp(0.0, 1.0)
}