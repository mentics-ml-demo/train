use std::{cmp::Ordering, collections::VecDeque, path::Path};

use anyhow::bail;
use shared_types::{util::now, *};
use series_store::*;
use kv_store::*;

use crate::{make_trainer, train::trainer::Trainer, TheAutodiffBackend};

pub struct DataMgr {
    _version: VersionType,
    series_labels: SeriesReader,
    series_events: SeriesReader,
    store: KVStore,
    trainer: Trainer<TheAutodiffBackend>,
}

pub async fn make_mgr<P: AsRef<Path>>(version: VersionType, path: P) -> anyhow::Result<DataMgr> {
    let series_labels = SeriesReader::new_topic(StdoutLogger::boxed(), &Topic::new("label", "SPY", "notify"))?;
    let series_events = SeriesReader::new_topic(StdoutLogger::boxed(), &Topic::new("raw", "SPY", "quote"))?;
    let store = KVStore::new(version).await?;
    let trainer = make_trainer(path)?;

    Ok(DataMgr::new(version, series_labels, series_events, store, trainer))
}

impl DataMgr {
    pub fn new(version: VersionType, series_labels: SeriesReader, series_events: SeriesReader, store: KVStore, trainer: Trainer<TheAutodiffBackend>) -> Self { // -> anyhow::Result<Self> {
        // store.max_trained_event_id(version).await?;
        Self { _version: version, series_labels, series_events, store, trainer }
    }

    // Loop through label events
    //  for each label event, loop through raw events until arriving at the event_id of the label event
    pub async fn run(&mut self) -> anyhow::Result<()> {
        let mut count = 0;
        let max = 100;

        let mut first = true;
        // let mut labevent: LabelEvent = self.series_labels.read_into()?;
        let mut buf = BufferEvents::new(SERIES_SIZE, &self.series_events);
        loop {
            // self.series_labels.print_status()?;
            // self.series_events.print_status()?;

            let labevent: LabelEvent = self.series_labels.read_into()?;
            println!("Read label event: {:?}", labevent);
            if !self.series_events.valid_offset_ids(labevent.offset_from, labevent.offset_to)? {
                println!("Skipping label event with offset 0, event_id: {}", labevent.event_id);
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
            let new_label = labevent.label;

            let retrainers = self.store.train_loss(31).await?;
            let (event_ids, (offsets, (inputs, labels))): (Vec<_>, (Vec<_>, (Vec<_>, Vec<_>))) = retrainers.into_iter().map(|row| (row.0, (row.3, (row.5, row.6)))).unzip();
            // let inputs2 = inputs.into_iter().map(TryInto::<ModelInputFlat>::try_into).collect::<Result<Vec<_>,_>>().map_err(|e| anyhow::anyhow!("{:?}", e))?;
            let mut inputs2 = inputs.into_iter().map(vec_flat_to_input).collect::<Result<Vec<_>,_>>()?;
            inputs2.push(new_input);
            let mut labels2 = labels.into_iter().map(vec_flat_to_label).collect::<Result<Vec<_>,_>>()?;
            labels2.push(new_label.value);
            let num_inputs = inputs2.len();

            let batch_train_result = self.trainer.train_batch(inputs2, labels2)?;
            println!("Trained {} events. Train result: {:?}", num_inputs, batch_train_result);
            let train_result = batch_train_result.last().unwrap().to_owned();
            self.store(labevent.event_id, labevent.offset_from, train_result, new_input, new_label).await?;
            // TODO: update the other events in trained store with new loss values
            self.series_labels.commit()?;

            count += 1;
            if count > max {
                self.trainer.save_model()?;
                println!("TODO: train debug stopping");
                break;
            }
        }
        Ok(())
    }

    async fn store(&self, event_id: EventId, offset: OffsetId, train: TrainType, arr: ModelInput, label: Label) -> anyhow::Result<()> {
        let timestamp = now();
        let to_store = TrainStored {
            event_id, timestamp,
            partition: PARTITION, offset,
            train: Train { loss: train }, input: arr, label
        };
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
        let mut count = 0;
        loop {
            let event: QuoteEvent = self.series_events.read_into_event()?;
            count += 1;
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
                    println!("Found matching event_id in events series {}, len: {}, after reading {count} events", event_id, self.events.len());
                    return Ok(&self.events);
                },
                Ordering::Greater => {
                    // Error, this shouldn't happen
                    bail!("event_id {event_id} was missing in event stream");
                },
            }
        }
    }
}

fn series_to_input<'a, T>(v: T) -> anyhow::Result<[[f32; NUM_FEATURES]; SERIES_SIZE]>
where T: IntoIterator<Item = &'a QuoteEvent> {
    let mut iter = v.into_iter();
    let mut arr = new_input();
    for [bid, ask] in arr.iter_mut() {
        if let Some(q) = iter.next() {
            (*bid, *ask) = (q.bid, q.ask);
        } else {
            bail!("series_to_input: iterator had insufficient items");
        }
    }

    if iter.next().is_some() {
        bail!("series_to_input: iterator had extra items");
    }

    let [base_bid, base_ask] = arr[SERIES_SIZE - 1];
    for [bid, ask] in arr.iter_mut() {
        *bid = adjust(base_bid / *bid);
        *ask = adjust(base_ask / *ask);
    }
    // TODO: remove this assert after test
    for [a, b] in arr {
        assert!(a > 0.0);
        assert!(b > 0.0);
    }
    Ok(arr)
}

fn adjust(x: f32) -> f32 {
    (x - 0.5).clamp(0.0, 1.0)
}

fn vec_flat_to_input(v: Vec<ModelFloat>) -> anyhow::Result<ModelInput> {
    Ok(to_model_input(TryInto::<ModelInputFlat>::try_into(v).map_err(|e| anyhow::anyhow!("{:?}", e))?))
}

fn vec_flat_to_label(v: Vec<ModelFloat>) -> anyhow::Result<LabelType> {
    Ok(TryInto::<LabelType>::try_into(v).map_err(|e| anyhow::anyhow!("{:?}", e))?)
}

