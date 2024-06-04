use std::{cmp::Ordering, collections::VecDeque};

use anyhow::bail;
use futures::future::join_all;

use shared_types::{*, convert::*, util::now};
use series_store::*;
use kv_store::*;

use crate::{artifacts_dir, convert::*, train::trainer::{make_trainer, Trainer}, TheAutodiffBackend};

pub struct DataMgr {
    _version: VersionType,
    series_labels: SeriesReader,
    series_events: SeriesReader,
    store: KVStore,
    trainer: Trainer<TheAutodiffBackend>,
}

pub async fn make_mgr(version: VersionType, reset: bool) -> anyhow::Result<DataMgr> {
    let series_labels = SeriesReader::new_topic2(StdoutLogger::boxed(), "train", &Topic::new("label", "SPY", "notify"), reset)?;
    let series_events = SeriesReader::new_topic2(StdoutLogger::boxed(), "train", &Topic::new("raw", "SPY", "quote"), reset)?;
    let store = KVStore::new(version).await?;
    let trainer = make_trainer(reset)?;

    let mgr = DataMgr::new(version, series_labels, series_events, store, trainer);

    if reset {
        println!("**** RESETTING TRAIN DATA ****");
        println!("  resetting offsets");
        mgr.reset_label_offset()?;
        println!("  deleting train data");
        mgr.reset_train_data().await?;
        let path = artifacts_dir()?;
        println!("  deleting model artifacts: {:?}", &path);
        std::fs::remove_dir_all(&path)?;
    }

    Ok(mgr)
}

impl DataMgr {
    pub fn new(version: VersionType, series_labels: SeriesReader, series_events: SeriesReader, store: KVStore, trainer: Trainer<TheAutodiffBackend>) -> Self { // -> anyhow::Result<Self> {
        // store.max_trained_event_id(version).await?;
        Self { _version: version, series_labels, series_events, store, trainer }
    }

    pub fn reset_label_offset(&self) -> anyhow::Result<()> {
        self.series_labels.reset_offset()
    }

    pub async fn reset_train_data(&self) -> anyhow::Result<()> {
        self.store.reset_train_data().await?;
        Ok(())
    }

    // Loop through label events
    //  for each label event, loop through raw events until arriving at the event_id of the label event
    pub async fn run(&mut self, run_count: usize) -> anyhow::Result<()> {
        let mut count = 0;
        let max = run_count;

        let mut first = true;
        // let mut labevent: LabelEvent = self.series_labels.read_into()?;
        let mut prev_labevent = LabelEvent::default();
        let mut buf = BufferEvents::new(SERIES_SIZE, &self.series_events);
        loop {
            let display_result = count % 100 == 0;
            let timestamp = now();
            // self.series_labels.print_status()?;
            // self.series_events.print_status()?;

            let labevent: LabelEvent = self.series_labels.read_into()?;
            println!("Read label event: {:?}", labevent);
            if !self.series_events.valid_offset_ids(labevent.offset_from, labevent.offset_to)? {
                println!("Skipping label event with offset 0, event_id: {}", labevent.event_id);
                continue;
            }

            if labevent.event_id == prev_labevent.event_id {
                println!("Found duplicate label event. Skipping:\n  {:?}\n  {:?}", labevent, prev_labevent);
                continue;
            }

            let target_position = labevent.offset_from - SERIES_LENGTH + 1;
            if first {
                println!("First time through loop, seeking to label event offset_from {} - num_series {} = {}, event_id: {}", labevent.offset_from, SERIES_LENGTH, target_position, labevent.event_id);
                self.series_events.seek(target_position)?;
                self.series_events.print_status()?;
                first = false;
            }

            let data = buf.read_to(labevent.event_id)?;
            if data.len() != SERIES_SIZE {
                bail!("data wrong length: {} != {SERIES_LENGTH}", data.len());
            }
            assert!(data.back().unwrap().event_id == labevent.event_id);
            let new_input = series_to_input(data)?;
            let new_label = labevent.label.clone(); // added clone to deal with checking for duplicates

            let mut retrainers = RetrainerData::new(self.store.train_top_full(63).await?)?;
            let ret_top_count = retrainers.event_ids.len();
            let ret2 = RetrainerData::new(self.store.train_oldest_full(32).await?)?;
            let ret_oldest_count = ret2.event_ids.len();
            retrainers.append(ret2);
            let RetrainerData { event_ids, losses, mut inputs, mut labels } = retrainers;

            inputs.push(new_input);
            labels.push(new_label.value);

            let batch_train_result = self.trainer.train_batch(inputs, labels, display_result)?;
            // println!("Trained {} events. Train result: {:?}", num_inputs, batch_train_result);
            let new_loss = batch_train_result.last().unwrap().to_owned();
            self.store_train_result(labevent.event_id, timestamp, labevent.offset_from, new_input, new_label, new_loss).await?;

            let futures = (0..event_ids.len()).map(|i| {
                self.store.train_loss_update(event_ids[i], timestamp, losses[i], batch_train_result[i])
            });
            let results = join_all(futures).await;
            let errors = results.into_iter()
                .filter(|result| result.is_err())
                .map(|result| {
                    println!("Error updating loss for retrainer: {:?}", result);
                    result.unwrap_err()
                }).collect::<Vec<_>>();
            if !errors.is_empty() {
                bail!("Errors updating loss for retrainers: {}", errors.len());
            }

            if display_result {
                let mut event_ids_sorted = event_ids.clone();
                event_ids_sorted.sort_unstable();
                println!("Top EventIds being processed this time {} top, {} oldest: {:?}", ret_top_count, ret_oldest_count, event_ids_sorted);
                self.trainer.save_model()?;
            }

            self.series_labels.commit()?;

            println!("Trained for label event_id: {:?}", labevent.event_id);
            // self.series_labels.print_status()?;
            prev_labevent = labevent;

            count += 1;
            if count >= max {
                self.trainer.save_model()?;
                println!("TODO: train debug stopping");
                break;
            }
        }
        Ok(())
    }

    async fn store_train_result(&self, event_id: EventId, timestamp: Timestamp, offset: OffsetId, arr: ModelInput, label: Label, loss: TrainResultType) -> anyhow::Result<()> {
        let train = TrainStored {
            event_id, timestamp,
            partition: PARTITION, offset,
            input: arr, label
        };
        self.store.train_store(train).await?;

        let train_loss = TrainLossStored { event_id, timestamp, loss };
        self.store.train_loss_store(train_loss).await
    }
}

struct RetrainerData {
    event_ids: Vec<u64>,
    losses: Vec<f32>,
    inputs: Vec<[[f32; 6]; 1024]>,
    labels: Vec<[f32; 8]>
}

impl RetrainerData {
    fn new(retrainers: Vec<TrainFull>) -> anyhow::Result<RetrainerData> {
        let (event_ids, (inputs, (labels, losses))): (Vec<_>, (Vec<_>, (Vec<_>, Vec<_>))) =
                retrainers.into_iter().map(|row| (row.event_id, (row.input_flat, (row.label_flat, row.loss)))).unzip();
        let inputs = inputs.into_iter().map(vec_flat_to_input).collect::<Result<Vec<_>,_>>()?;
        let labels = labels.into_iter().map(vec_flat_to_label).collect::<Result<Vec<_>,_>>()?;
        Ok(RetrainerData { event_ids, losses, inputs, labels })
    }

    fn append(&mut self, mut to_append: RetrainerData) {
        self.event_ids.append(&mut to_append.event_ids);
        self.losses.append(&mut to_append.losses);
        self.inputs.append(&mut to_append.inputs);
        self.labels.append(&mut to_append.labels);
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
        // let mut count = 0;
        loop {
            let event: QuoteEvent = self.series_events.read_into_event()?;
            // count += 1;
            // println!("Read event at offset {}, event_id: {}", event.offset, event.event_id);
            if self.events.len() == self.target_length {
                let _ = self.events.pop_front();
                // let front = self.events.pop_front();
                // println!("pop_front called at offset {}", front.unwrap().offset);
            }
            let found_event_id = event.event_id;
            self.events.push_back(event);

            match found_event_id.cmp(&event_id) {
                Ordering::Less => (), // Keep going
                Ordering::Equal => {
                    // println!("Found matching event_id in events series {}, len: {}, after reading {count} events", event_id, self.events.len());
                    return Ok(&self.events);
                },
                Ordering::Greater => {
                    // Error, this shouldn't happen
                    bail!("event_id {event_id} was missing in event stream, found {found_event_id}");
                },
            }
        }
    }
}
