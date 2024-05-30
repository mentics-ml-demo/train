use std::{collections::VecDeque, path::Path};

use anyhow::bail;
use shared_types::{util::now, *};
use series_store::*;
use kv_store::*;

use crate::{make_trainer, train::trainer::Trainer, TheAutodiffBackend};

pub struct DataMgr {
    version: VersionType,
    series_labels: SeriesReader,
    series_events: SeriesReader,
    store: KVStore,
    trainer: Trainer<TheAutodiffBackend>,
}

pub async fn make_mgr<P: AsRef<Path>>(path: P) -> anyhow::Result<DataMgr> {
    let series_labels = SeriesReader::new_topic(StdoutLogger::boxed(), &Topic::new("label", "SPY", "notify"))?;
    let series_events = SeriesReader::new_topic(StdoutLogger::boxed(), &Topic::new("raw", "SPY", "quote"))?;
    let store = KVStore::new(CURRENT_VERSION).await?;
    let trainer = make_trainer(path)?;

    Ok(DataMgr::new(CURRENT_VERSION, series_labels, series_events, store, trainer))
}

impl DataMgr {
    pub fn new(version: VersionType, series_labels: SeriesReader, series_events: SeriesReader, store: KVStore, trainer: Trainer<TheAutodiffBackend>) -> Self { // -> anyhow::Result<Self> {
        // store.max_trained_event_id(version).await?;
        Self { version, series_labels, series_events, store, trainer }
    }

    // Loop through label events
    //  for each label event, loop through raw events until arriving at the event_id of the label event
    pub async fn run(&mut self) -> anyhow::Result<()> {
        self.series_labels.print_status()?;
        self.series_events.print_status()?;

        // TODO: series_events needs to seek to the offset of the first label event it's going to work off...
        // just commiting offset in series_events will be too far forward when we start again.

        // let mut labevent: LabelEvent = self.series_labels.read_into()?;
        let mut buf = BufferEvents::new(SERIES_LENGTH, &self.series_events);
        loop {
            let labevent: LabelEvent = self.series_labels.read_into()?;
            println!("Read label event: {:?}", labevent);
            if !self.series_events.valid_offset_id(labevent.offset)? {
                println!("Skipping label event with offset 0, event_id: {}", labevent.event_id);
                continue;
            }

            let data = buf.read_to(labevent.event_id)?;
            if data.len() < SERIES_LENGTH {
                bail!("data too short: {}", data.len());
            } else if data.len() > SERIES_LENGTH {
                bail!("data too long: {}", data.len());
            }
            // if let Some(data) = buf.read_to(labevent.event_id) {
            assert!(data.back().unwrap().event_id == labevent.event_id);
            let arr = quotes_to_arrays(data)?;
            let train_result = self.trainer.train_full(arr, labevent.label.value)?;
            self.store(labevent.event_id, train_result).await?;
            self.series_labels.commit()?;
            if true { break };
        }
        Ok(())

        // let mut trained_event_id = self.store.max_trained_event_id(self.version).await?.unwrap_or(0);
        // let mut count = 0;

        // self.series.read()
        // let buffer = BufferResults::new(&self.store, event_id);

        // loop {
        //     let labeleds = self.store.labeled_next(self.version, trained_event_id).await?;
        //     if labeleds.is_none() {
        //         break;
        //     }

        //     let labeled = labeleds.unwrap();
        //     self.series.seek(&self.data_topic, labeled.partition, Offset::Offset(labeled.offset - SERIES_LENGTH as i64 + 1))?;
        //     println!("{}: read {SERIES_LENGTH} events at offset: {}", labeled.event_id, labeled.offset);
        //     let quotes: Vec<QuoteEvent> = self.series.read_count_into(SERIES_LENGTH)?;
        //     assert!(quotes.last().unwrap().event_id == labeled.event_id);
        //     let arr = quotes_to_arrays(quotes);
        //     self.trainer.train_full(arr, labeled.label.value)?;
        //     trained_event_id = labeled.event_id;

        //     count += 1;
        //     if count > 1 { break; }
        // }
        // self.trainer.save_model()?;
        // Ok(count)
    }

    async fn store(&self, event_id: EventId, train: TrainType) -> anyhow::Result<()> {
        let timestamp = now();
        let to_store = TrainStored { event_id, timestamp, train: Train { loss: train } };
        self.store.train_store(to_store).await
    }
}

struct BufferEvents<'a> {
    target_length: usize,
    series_events: &'a SeriesReader,
    events: VecDeque<QuoteEvent>,
}

impl<'a> BufferEvents<'a> {
    pub fn new(target_length: usize, series_events: &'a SeriesReader) -> Self {
        Self { target_length, series_events, events: VecDeque::new() }
    }
    // TODO: do we need to seek to the right place? or default use stored offset?

    pub fn read_to(&mut self, event_id: EventId) -> anyhow::Result<&VecDeque<QuoteEvent>> {
        // &'a Vec<QuoteEvent>
        loop {
            let event: QuoteEvent = self.series_events.read_into_event()?;
            // println!("Read event at offset {}", event.offset);
            if self.events.len() == self.target_length {
                let front = self.events.pop_front();
                println!("pop_front called at offset {}", front.unwrap().offset);
            }
            let found_event_id = event.event_id;
            self.events.push_back(event);
            if found_event_id == event_id {
                println!("Found matching event_id in events series {}, len: {}", event_id, self.events.len());
                return Ok(&self.events);
            } else if found_event_id > event_id {
                bail!("event_id {event_id} was missing in event stream");
            }
        }
    }

    // pub fn next<'b>(&'b mut self, event_id: EventId) -> anyhow::Result<Option<Labeled>> {
    //     if self.labels.is_empty() {
    //         self.fetch_more_events(event_id);
    //     }
    //     while let Some(lab) = self.labels.pop_front() {
    //         if lab.event_id == event_id {
    //             let events = self.series_events.collect_while(|event: &QuoteEvent| {
    //                 event.event_id < lab.event_id
    //             })?;
    //             self.events.extend(events);
    //             self.events.back().map(|x| x.event_id)
    //             return Ok(Some(lab));
    //         } else if lab.event_id > event_id {
    //             // Revet the pop because we're out of order somehow
    //             self.labels.push_front(lab);
    //             return Ok(None);
    //         }
    //     }
    //     bail!("no more labels")
    // }

    // async fn fetch_more_events(&mut self, event_id: EventId) -> anyhow::Result<()> {
    //     self.events.extend(self.store.labeled_next(event_id, 10).await?);
    //     Ok(())
    // }
}

fn quotes_to_arrays<'a, T>(v: T) -> anyhow::Result<[[f32; NUM_FEATURES]; SERIES_LENGTH]>
where T: IntoIterator<Item = &'a QuoteEvent> {
    let mut iter = v.into_iter();
    let mut arr = [[0f32; NUM_FEATURES]; SERIES_LENGTH];
    for i in 0..SERIES_LENGTH {
        if let Some(q) = iter.next() {
            arr[i][0] = q.bid;
            arr[i][1] = q.ask;
        } else {
            bail!("quotes_to_arrays: iterator had insufficient items {i}");
        }
    }
    if !iter.next().is_none() {
        bail!("quotes_to_arrays: iterator had extra items");
    }
    let base = arr[SERIES_LENGTH - 1];
    for i in 0..SERIES_LENGTH {
        arr[i][0] = adjust(base[0] / arr[i][0]);
        arr[i][1] = adjust(base[1] / arr[i][1]);
    }
    Ok(arr)
}

fn adjust(x: f32) -> f32 {
    (x - 0.5).clamp(0.0, 1.0)
}