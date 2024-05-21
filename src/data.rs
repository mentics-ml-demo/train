use shared_types::*;
use series_store::*;
use kv_store::*;

use crate::Trainer;

pub struct DataMgr {
    version: VersionType,
    series: SeriesReader,
    store: KVStore,
    data_topic: Topic,
    trainer: Trainer,
}

impl DataMgr {
    pub async fn new(version: VersionType, mut series: SeriesReader, store: KVStore, topic: Topic, trainer: Trainer) -> anyhow::Result<Self> {
        store.max_trained_event_id(version).await?;
        series.subscribe(&topic, Offset::Beginning)?;

        // let msgs = series.read_count(SERIES_LENGTH)?;
        // println!("count: {}", msgs.len());
        // println!("first: {:?}", msg_to::<QuoteEvent>(msgs.first().unwrap())?);
        Ok(Self { version, series, store, data_topic: topic, trainer })
    }

    pub async fn run(&self) -> anyhow::Result<u64> {
        let mut trained_event_id = self.store.max_trained_event_id(self.version).await?.unwrap_or(0);
        let mut count = 0;
        loop {
            let labeleds = self.store.labeled_next(self.version, trained_event_id).await?;
            if labeleds.is_none() {
                break;
            }

            let labeled = labeleds.unwrap();
            self.series.seek(&self.data_topic, labeled.partition, Offset::Offset(labeled.offset - SERIES_LENGTH as i64 + 1))?;
            let quotes: Vec<QuoteEvent> = self.series.read_count_into(SERIES_LENGTH)?;
            assert!(quotes.last().unwrap().event_id == labeled.event_id);
            self.trainer.train(quotes, labeled.label);
            trained_event_id = labeled.event_id;

            count += 1;
            if count > 2 { break; }
        }
        Ok(count)
    }
}
