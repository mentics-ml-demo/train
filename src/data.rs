use std::cmp::max;
use std::sync::Arc;
use anyhow::bail;
use async_scoped::TokioScope;
use burn::nn::loss::MseLoss;
use burn::optim::AdamWConfig;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::Device;

use chrono_util::to_datetime;
use convert::series_to_input;
use shared_burn::model::*;
use shared_burn::tensor1_to_vec;
use shared_types::*;
use data_info::*;
use series_store::*;
use kv_store::*;
use chrono_util::now;
use label::LabelEvent;
use paths::artifacts_dir;
use shared_burn::model_persist::*;
use shared_burn::TheAutodiffBackend;
use stored::{InputStored, LabelTypeStored, TrainStored, TrainStoredWithLabel};

use crate::train::trainer::MseLossCalc;
use crate::train::trainer::{load_train_config, save_train_config, TheTrainingConfig};
use crate::{events_window::EventsWindow, train::trainer::Trainer};

pub struct DataMgr<B:AutodiffBackend> {
    _version: VersionType,
    device: B::Device,
    series_labels: Arc<SeriesReader>,
    series_events: SeriesReader,
    store: Arc<KVStore>,
    trainer: Trainer<B, TheModel<B>, MseLossCalc<B>>,
}

pub async fn make_mgr(version: VersionType, reset: bool) -> anyhow::Result<DataMgr<TheAutodiffBackend>> {
    let series_labels = SeriesReader::new_topic2("train", &Topic::new("label", "SPY", "notify"), reset)?;
    let series_events = SeriesReader::new_topic2("train", &Topic::new("raw", "SPY", "quote"), reset)?;
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

    let device = burn::backend::wgpu::WgpuDevice::default();
    let trainer = make_trainer(&device, reset)?;
    let mgr = DataMgr::new(version, device, series_labels, series_events, store, trainer);
    Ok(mgr)
}

impl<B:AutodiffBackend> DataMgr<B> {
    pub fn new(version: VersionType, device: B::Device, series_labels: SeriesReader, series_events: SeriesReader, store: KVStore, trainer: Trainer<B, TheModel<B>, MseLossCalc<B>>) -> Self {
        // store.max_trained_event_id(version).await?;
        Self { _version: version, device, series_labels: Arc::new(series_labels), series_events, store: Arc::new(store), trainer }
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
        let (low, _) = self.series_events.fetch_watermarks()?;
        let minimum_offset_id = low + SERIES1_LENGTH;
        let mut labevent: LabelEvent;
        loop {
            labevent = self.read_next_label()?;
            if labevent.offset_from >= minimum_offset_id {
                break;
            } else {
                println!("skipped labevent because offset_from < minimum_offset_id: {} > {}", labevent.offset_from, minimum_offset_id);
            }
        }
        let target_position = max(0, labevent.offset_from - SERIES1_LENGTH + 1);
        println!("First time through loop, seeking to label event offset_from {} - num_series {} = {}, event_id: {}", labevent.offset_from, SERIES1_LENGTH, target_position, labevent.event_id);
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

    async fn update_losses(store: Arc<KVStore>, timestamp: Timestamp, event_ids: Vec<EventId>, new_losses: Vec<f32>) -> bool {
        let expected_len = event_ids.len() as u64;
        let rows_modified = err_return!(store.update_losses(timestamp, event_ids, new_losses).await, "Error updating losses: {}", false);
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
        let mut timestamp = now();
        let mut buf = EventsWindow::new(SERIES1_SIZE); // , &self.series_events
        let mut prev_labevent = self.seek_to_start()?;

        let (train, event_ids, new_losses) = self.train_event(timestamp, &mut buf, &prev_labevent, false)?;
        Self::store_train(self.store.clone(), self.series_labels.clone(), train).await;
        if !event_ids.is_empty() {
            Self::update_losses(self.store.clone(), timestamp, event_ids, new_losses).await;
        }
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
                    timestamp = now();

                    let occasional = count % 100 == 0;

                    let labevent = err_return!(self.read_next_label_no_dupe(&prev_labevent), "Error reading next label: {}", false);
                    let (train, event_ids, new_losses) = err_return!(self.train_event(timestamp, &mut buf, &labevent, occasional), "Error training event: {}", false);
                    scope.spawn(Self::store_train(self.store.clone(), self.series_labels.clone(), train));
                    scope.spawn(Self::update_losses(self.store.clone(), timestamp, event_ids, new_losses));

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

    fn train_event(&mut self, timestamp: Timestamp, buf: &mut EventsWindow, labevent: &LabelEvent, occasional: bool) -> anyhow::Result<(TrainStored, Vec<EventId>, Vec<ModelFloat>)> {
        let (new_input, new_label) = self.read_next_input(buf, labevent)?;

        let retrainers = futures::executor::block_on(self.store.retrainers(15, 16))?;
        let retrainer_data = RetrainerData::new(retrainers);
        let RetrainerData { event_ids, mut inputs, mut labels } = retrainer_data;
        inputs.push(new_input);
        labels.push(new_label);
        // NOTE: later it assumes that event_ids and losses_out do not include the new one, so the lengths will be different by one.
        let batch_size = inputs.len();

        let input: ModelInput<B> = inputs_to_device(inputs, &self.device);
        let expected: ModelOutput<B> = outputs_to_device(labels, &self.device);
        let (new_losses, new_outputs) = self.trainer.train(input, expected)
                .map_err(|e| anyhow::anyhow!("train_batch error: {}", e))?;
        // println!("Trained {} events. Train result: {:?}", num_inputs, batch_train_result);
        let losses_out = tensor1_to_vec(new_losses);
        let new_loss = losses_out.last().unwrap().to_owned();
        let new_output = tensor1_to_vec(new_outputs.slice([batch_size-1..batch_size]).squeeze(0)).try_into()
                .map_err(|_| anyhow::anyhow!("Error extracting new output"))?;
        // let new_output = new_outputs.last().unwrap().to_owned();

        if occasional {
            println!("occasional start");
            let mut event_ids_sorted = event_ids.clone();
            event_ids_sorted.sort_unstable();
            println!("EventIds being processed this time {} retrainers: {:?}", event_ids.len(), event_ids_sorted);
            self.trainer.save_model()?;
            println!("occasional end");
        }

        println!("Trained for label {}: {:?}", to_datetime(labevent.timestamp), labevent);

        let train = TrainStored { event_id: labevent.event_id, timestamp, offset: labevent.offset_from, loss: new_loss, input: new_input, output: new_output };
        Ok((train, event_ids, losses_out))
    }

    fn read_next_input(&mut self, buf: &mut EventsWindow, labevent: &LabelEvent) -> Result<(InputRaw, LabelType), anyhow::Error> {
        let data = buf.read_to(labevent.event_id, &self.series_events)?;
        if data.len() != SERIES1_SIZE {
            bail!("data wrong length: {} != {SERIES1_LENGTH}", data.len());
        }
        assert!(data.back().unwrap().event_id == labevent.event_id);
        let new_input = series_to_input(data)?;
        Ok((new_input, labevent.label))
    }
}

struct RetrainerData {
    event_ids: Vec<EventId>,
    inputs: Vec<InputStored>,
    labels: Vec<LabelTypeStored>
}

impl RetrainerData {
    fn new(retrainers: Vec<TrainStoredWithLabel>) -> RetrainerData {
        let (event_ids, (inputs, labels)): (Vec<_>, (Vec<_>, Vec<_>)) =
            retrainers.into_iter().map(|train| (train.event_id, (train.input, train.label))).unzip();
        RetrainerData { event_ids, inputs, labels }
    }
}

// pub fn make_trainer<B:AutodiffBackend,M>(device: &B::Device, new_model: bool) -> anyhow::Result<Trainer<B,M,MseLossCalc<B>>>
// where M: Model + AutodiffModule<B>
// {
//     let (train_config, model) = if new_model {
//         let model_config = TheModelConfig::new_config();
//         save_model_config(&model_config)?;
//         let model = model_config.init::<TheAutodiffBackend>(device);
//         let save_model = save_model(&model)?;
//         let train_config = TheTrainingConfig::new(model_config, AdamWConfig::new());
//         save_train_config(&train_config)?;
//         (train_config, model)
//     } else {
//         (load_train_config()?, load_model(device)?)
//         // let model_path = artifacts_dir()?.join("model.mpk");
//         // if model_path.try_exists()? {
//         //     println!("Loading model... (from {:?})", &model_path);
//         //     let record = CompactRecorder::new().load(model_path, &device).with_context(|| "Error loading model")?;
//         //     model = config.model.init::<B>(&device).load_record(record);
//         //     println!("  model loaded.")
//         // }
//     };

//     let loss_calc = MseLossCalc(MseLoss::new());
//     Trainer::new(train_config, model, loss_calc)
// }

type B = TheAutodiffBackend;
type ModelType = TheModel<B>;
type LossType = MseLossCalc<B>;
type TrainerType = Trainer<B,ModelType,LossType>;

pub fn make_trainer(device: &Device<B>, new_model: bool) -> anyhow::Result<TrainerType> {
    let (train_config, model) = if new_model {
        let model_config = TheModelConfig::new_config();
        save_model_config(&model_config)?;
        let model = model_config.init::<TheAutodiffBackend>(device);
        save_model(&model)?;
        let train_config = TheTrainingConfig::new(model_config, AdamWConfig::new());
        save_train_config(&train_config)?;
        (train_config, model)
    } else {
        (load_train_config()?, load_model(device)?)
        // let model_path = artifacts_dir()?.join("model.mpk");
        // if model_path.try_exists()? {
        //     println!("Loading model... (from {:?})", &model_path);
        //     let record = CompactRecorder::new().load(model_path, &device).with_context(|| "Error loading model")?;
        //     model = config.model.init::<B>(&device).load_record(record);
        //     println!("  model loaded.")
        // }
    };

    let loss_calc = MseLossCalc(MseLoss::new());
    Trainer::new(train_config, model, loss_calc)
}
