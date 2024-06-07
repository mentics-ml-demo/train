use std::sync::Arc;
use anyhow::bail;
use async_scoped::TokioScope;

use paths::artifacts_dir;
use shared_types::{*, util::now};
use series_store::*;
use kv_store::*;

use crate::{convert::*, events_window::EventsWindow, train::trainer::{make_trainer, Trainer}, TheAutodiffBackend};

pub struct DataMgr {
    _version: VersionType,
    series_labels: Arc<SeriesReader>,
    series_events: SeriesReader,
    store: Arc<KVStore>,
    trainer: Trainer<TheAutodiffBackend>,
}

pub async fn make_mgr(version: VersionType, reset: bool) -> anyhow::Result<DataMgr> {
    let series_labels = SeriesReader::new_topic2(StdoutLogger::boxed(), "train", &Topic::new("label", "SPY", "notify"), reset)?;
    let series_events = SeriesReader::new_topic2(StdoutLogger::boxed(), "train", &Topic::new("raw", "SPY", "quote"), reset)?;
    let store = KVStore::new(version).await?;

    if reset {
        println!("**** RESETTING TRAIN DATA ****");
        println!("  resetting offsets");
        series_labels.reset_offset()?;
        println!("  deleting train data");
        store.reset_train_data().await?;
        let path = artifacts_dir()?;
        println!("  deleting model artifacts: {:?}", &path);
        std::fs::remove_dir_all(&path)?;
    }

    let trainer = make_trainer(reset)?;
    let mgr = DataMgr::new(version, series_labels, series_events, store, trainer);
    Ok(mgr)
}

impl DataMgr {
    pub fn new(version: VersionType, series_labels: SeriesReader, series_events: SeriesReader, store: KVStore, trainer: Trainer<TheAutodiffBackend>) -> Self { // -> anyhow::Result<Self> {
        // store.max_trained_event_id(version).await?;
        Self { _version: version, series_labels: Arc::new(series_labels), series_events, store: Arc::new(store), trainer }
    }

    fn read_next_label(&self) -> anyhow::Result<LabelEvent> {
        let result = loop {
            let labevent: LabelEvent = self.series_labels.read_into()?;
            // if !self.series_events.valid_offset_ids(labevent.offset_from, labevent.offset_to)? {
            // TODO: it's expensive to fetch watermarks for every label event. is there another way to validate?
            if labevent.offset_from <= 0 || labevent.offset_to <= 0 {
                println!("Skipping label event with offset 0, event_id: {:?}", labevent);
                continue
            } else {
                println!("Read label event: {:?}", labevent);
                break labevent
            };
        };
        Ok(result)
    }

    fn read_next_label_no_dupe(&self, prev_labevent: &LabelEvent) -> anyhow::Result<LabelEvent> {
        let result = loop {
            let labevent = self.read_next_label()?;
            if labevent.event_id == prev_labevent.event_id {
                println!("Found duplicate label event. Skipping:\n  {:?}\n  {:?}", labevent, prev_labevent);
                continue
            } else {
                break labevent
            }
        };
        Ok(result)
    }

    fn seek_to_start(&self) -> anyhow::Result<LabelEvent> {
        let labevent = self.read_next_label()?;
        let target_position = labevent.offset_from - SERIES_LENGTH + 1;
        println!("First time through loop, seeking to label event offset_from {} - num_series {} = {}, event_id: {}", labevent.offset_from, SERIES_LENGTH, target_position, labevent.event_id);
        self.series_events.seek(target_position)?;
        self.series_events.print_status()?;
        Ok(labevent)
    }

    async fn store_train(store: Arc<KVStore>, series_labels: Arc<SeriesReader>, train: TrainStored) -> bool {
        let rows_modified = err_return!(store.train_store(train).await, "Error storing train: {}", false);
        if rows_modified != 1 {
            println!("Storing train did not return 1 modified row: {}", rows_modified);
        }
        err_return!(series_labels.commit(), "Error commiting series_labels: {}", false);
        true
    }

    async fn update_losses(store: Arc<KVStore>, event_ids: Vec<EventId>, new_losses: Vec<f32>) -> bool {
        let expected_len = event_ids.len() as u64;
        let rows_modified = err_return!(store.update_losses(event_ids, new_losses).await, "Error updating losses: {}", false);
        if rows_modified != expected_len {
            // NOTE: there's a chance the loss did not change, in which case this could happen
            println!("Updating losses did not return expected modified row count: {}", rows_modified);
        }
        true
    }

    // Loop through label events
    //  for each label event, loop through raw events until arriving at the event_id of the label event
    pub async fn run(&mut self, run_count: usize) -> anyhow::Result<()> {
        let max_iter_per_loop = 63;

        let mut buf = EventsWindow::new(SERIES_SIZE); // , &self.series_events
        let mut prev_labevent = self.seek_to_start()?;
        let (train, event_ids, new_losses) = self.train_event(&mut buf, &prev_labevent, false)?;
        Self::store_train(self.store.clone(), self.series_labels.clone(), train).await;
        Self::update_losses(self.store.clone(), event_ids, new_losses).await;
        let mut count = 1;

        loop {
            let (cont, _) = TokioScope::scope_and_block(|scope| {
                loop {
                    if count >= run_count {
                        if let Err(e) = self.trainer.save_model() {
                            println!("Error saving model after traniing {}", e);
                        }
                        println!("TODO: train debug stopping");
                        break false;
                    }

                    let occasional = count % 100 == 0;

                    let labevent = err_return!(self.read_next_label_no_dupe(&prev_labevent), "Error reading next label: {}", false);
                    let (train, event_ids, new_losses) = err_return!(self.train_event(&mut buf, &labevent, occasional), "Error training event: {}", false);
                    scope.spawn(Self::store_train(self.store.clone(), self.series_labels.clone(), train));
                    scope.spawn(Self::update_losses(self.store.clone(), event_ids, new_losses));

                    prev_labevent = labevent;

                    count += 1;

                    if count % 10 == 0 {
                        println!("len: {}", scope.len());
                        println!("remaining: {}", scope.remaining());
                    }

                    if count % max_iter_per_loop == 0 {
                        println!("Breaking inner loop to let threads catch up");
                        println!("len: {}", scope.len());
                        println!("remaining: {}", scope.remaining());
                        break true;
                    }
                }
            });
            if !cont {
                break;
            }
        }
        Ok(())
    }

    fn train_event(&mut self, buf: &mut EventsWindow, labevent: &LabelEvent, occasional: bool) -> anyhow::Result<(TrainStored, Vec<EventId>, Vec<ModelFloat>)> {
        let (new_input, new_label) = self.read_next_input(buf, labevent)?;

        let timestamp = now();
        let retrainers = futures::executor::block_on(self.store.retrainers(31, 32))?;
        let retrainer_data = RetrainerData::new(retrainers);
        let RetrainerData { event_ids, mut inputs, mut labels } = retrainer_data;
        inputs.push(new_input);
        labels.push(new_label.value);

        let (new_losses, new_outputs) = self.trainer.train_batch(inputs, labels, occasional)
                .map_err(|e| anyhow::anyhow!("train_batch error: {}", e))?;
        // println!("Trained {} events. Train result: {:?}", num_inputs, batch_train_result);
        let new_loss = new_losses.last().unwrap().to_owned();
        let new_output = new_outputs.last().unwrap().to_owned();

        if occasional {
            println!("occasional start");
            let mut event_ids_sorted = event_ids.clone();
            event_ids_sorted.sort_unstable();
            println!("EventIds being processed this time {} retrainers: {:?}", event_ids.len(), event_ids_sorted);
            self.trainer.save_model()?;
            println!("occasional end");
        }

        println!("Trained for label event_id: {:?}", labevent);

        let train = TrainStored { event_id: labevent.event_id, timestamp, offset: labevent.offset_from, loss: new_loss, input: new_input, output: new_output };
        Ok((train, event_ids, new_losses))
    }

    fn read_next_input(&mut self, buf: &mut EventsWindow, labevent: &LabelEvent) -> Result<([[f32; 6]; 1024], Label), anyhow::Error> {
        let data = buf.read_to(labevent.event_id, &self.series_events)?;
        if data.len() != SERIES_SIZE {
            bail!("data wrong length: {} != {SERIES_LENGTH}", data.len());
        }
        assert!(data.back().unwrap().event_id == labevent.event_id);
        let new_input = series_to_input(data)?;
        let new_label = labevent.label.clone();
        Ok((new_input, new_label))
    }
}

struct RetrainerData {
    event_ids: Vec<EventId>,
    inputs: Vec<ModelInput>,
    labels: Vec<LabelType>
}

impl RetrainerData {
    fn new(retrainers: Vec<TrainStoredWithLabel>) -> RetrainerData {
        let (event_ids, (inputs, labels)): (Vec<_>, (Vec<_>, Vec<_>)) =
            retrainers.into_iter().map(|train| (train.event_id, (train.input, train.label))).unzip();
        RetrainerData { event_ids, inputs, labels }
    }
}
