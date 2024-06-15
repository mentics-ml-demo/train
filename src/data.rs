use std::sync::Arc;
use std::sync::Mutex;
use async_scoped::TokioScope;
use burn::module::Module;
use burn::nn::loss::BinaryCrossEntropyLossConfig;
use burn::optim::AdamWConfig;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::Device;

use chrono_util::to_market_datetime;
use convert::series_to_input;
use itertools::izip;
use series::SeriesEvent;
use shared_burn::burn_device;
use shared_burn::model::*;
use shared_burn::tensor1_to_vec;
use shared_burn::tensor2_to_label_vec;
use shared_types::*;
use data_info::*;
use series_store::*;
use kv_store::*;
use chrono_util::now;
use label::LabelEvent;
use paths::artifacts_dir;
use shared_burn::TheAutodiffBackend;
use stored::{InputStored, LabelTypeStored, TrainStored, TrainStoredWithLabel};

use crate::train::trainer::BinaryCrossEntropyLossCalc;
use crate::train::trainer::{load_train_config, save_train_config, TheTrainingConfig};
use crate::{events_window::EventsWindow, train::trainer::Trainer};

pub struct DataMgr<B:AutodiffBackend> {
    _version: VersionType,
    device: B::Device,
    series_labels: Arc<SeriesReader>,
    series_events: SeriesReader,
    store: Arc<KVStore>,
    trainer: Trainer<B, TheModel<B>, BinaryCrossEntropyLossCalc<B>>,
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

    let device = burn_device();
    let trainer = make_trainer(&device, reset)?;
    let mgr = DataMgr::new(version, device, series_labels, series_events, store, trainer);
    Ok(mgr)
}

impl<B:AutodiffBackend> DataMgr<B> {
    pub fn new(version: VersionType, device: B::Device, series_labels: SeriesReader, series_events: SeriesReader, store: KVStore,
            trainer: Trainer<B, TheModel<B>, BinaryCrossEntropyLossCalc<B>>) -> Self {
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
                // println!("Read label event: {:?}", labevent);
                break labevent
            };
        };
        Ok(result)
    }

    // fn read_next_label_no_dupe(&self, prev_labevent: &LabelEvent) -> anyhow::Result<LabelEvent> {
    //     let result = loop {
    //         let labevent = self.read_next_label()?;
    //         if labevent.event_id == prev_labevent.event_id {
    //             println!("Found duplicate label event. Skipping:\n  {:?}\n  {:?}", labevent, prev_labevent);
    //             continue
    //         } else {
    //             break labevent
    //         }
    //     };
    //     Ok(result)
    // }

    // fn seek_to_start(&self) -> anyhow::Result<LabelEvent> {
    //     let (low, _) = self.series_events.fetch_watermarks()?;
    //     let minimum_offset_id = low + SERIES1_LENGTH;
    //     let mut labevent: LabelEvent;
    //     loop {
    //         labevent = self.read_next_label()?;
    //         if labevent.offset_from >= minimum_offset_id {
    //             break;
    //         } else {
    //             println!("skipped labevent because offset_from < minimum_offset_id: {} > {}", labevent.offset_from, minimum_offset_id);
    //         }
    //     }
    //     let target_position = max(0, labevent.offset_from - SERIES1_LENGTH + 1);
    //     println!("First time through loop, seeking to label event offset_from {} - num_series {} = {}, event_id: {}", labevent.offset_from, SERIES1_LENGTH, target_position, labevent.event_id);
    //     self.series_events.seek(target_position)?;
    //     self.series_events.print_status()?;
    //     Ok(labevent)
    // }

    async fn store_trains(store: Arc<KVStore>, series_labels: Arc<SeriesReader>, cur_time: Timestamp,
            labevents: Vec<LabelEvent>, inputs: Vec<InputRaw>, outputs: Vec<LabelType>, losses: Vec<ModelFloat>) -> bool {
        let mut trains = Vec::new();
        for (labevent, input, output, loss) in izip!(labevents, inputs, outputs, losses) {
            trains.push(TrainStored {
                event_id: labevent.event_id, timestamp: cur_time, offset: labevent.offset_from,
                loss, input, output
            });
        }

        let rows_modified = err_return!(store.store_trains(trains).await, "Error storing train: {}", false);
        // println!("Batch update for train_store modified {} rows", rows_modified);
        if rows_modified < 1 {
            println!("Storing train returned < 1 modified rows: {}", rows_modified);
        }

        err_return!(series_labels.commit(), "Error commiting series_labels: {}", false);
        true
    }

    async fn update_retrainers(store: Arc<KVStore>, cur_time: Timestamp, event_ids: Vec<EventId>, outputs: Vec<LabelType>, losses: Vec<f32>) -> bool {
        let expected_len = event_ids.len() as u64;
        let rows_modified = err_return!(store.update_retrainers(cur_time, event_ids, outputs, losses).await, "Error updating retrainers: {}", false);
        if rows_modified != expected_len {
            // NOTE: there's a chance the loss did not change, in which case this could happen
            println!("Updating losses did not return expected modified row count: {}", rows_modified);
        }
        true
    }

    // Loop through label events
    //  for each label event, loop through raw events until arriving at the event_id of the label event
    pub async fn run(&mut self, run_count: usize) -> anyhow::Result<()> {
        let batch_counts = TrainSourceCounts::new(32, 32, 64);
        let max_iter_per_loop = 64;
        let mut cur_time = now();
        let mut buf = EventsWindow::new(SERIES1_SIZE); // , &self.series_events
        let mut iter_count = 1; // 1 instead of 0 to avoid occasional on first iteration
        let mut occasional = false;
        let mut newtrained_count = 0;
        let mut retrained_count = 0;

        loop {
            let (cont, _) = TokioScope::scope_and_block(|scope| {
                loop {
                    if iter_count > run_count {
                        println!("Ending for count: {}", run_count);
                        err_return!(self.series_labels.commit(), "Error commiting series_labels: {}", false);
                        let _ = Self::save_model_arc(self.trainer.model.clone());
                        println!("TODO: train debug stopping");
                        break false;
                    }

                    // TODO: do it by time, not iter count
                    occasional = iter_count % 10 == 0;
                    cur_time = now();

                    let mut labevents: Vec<LabelEvent> = Vec::with_capacity(batch_counts.new);
                    while labevents.len() < batch_counts.new {
                        let labevent = err_return!(self.read_next_label(), "Error reading label event {}", false);
                        if let Some(prev_labevent) = labevents.last() {
                            if labevent.event_id == prev_labevent.event_id {
                                println!("Found duplicate label event. Skipping:\n  {:?}\n  {:?}", labevent, prev_labevent);
                                continue
                            }
                        }

                        labevents.push(labevent);
                    }
                    newtrained_count += labevents.len();

                    let (event_ids, inputs, mut outputs, mut losses) =
                        err_return!(self.train_event(&mut buf, &labevents, &batch_counts), "Error training event: {}", false);
                    if inputs.is_empty() {
                        continue
                    }
                    let retrainer_count = event_ids.len();
                    retrained_count += retrainer_count;
                    // TODO: could avoid a vec alloc here by compile time array sizing, but... wait to see if profiling says worth it.
                    let outputs_new = outputs.split_off(retrainer_count);
                    let losses_new = losses.split_off(retrainer_count);

                    scope.spawn(Self::store_trains(self.store.clone(), self.series_labels.clone(), cur_time, labevents, inputs, outputs_new, losses_new));
                    if retrained_count > 0 {
                        scope.spawn(Self::update_retrainers(self.store.clone(), cur_time, event_ids.clone(), outputs, losses));
                    }

                    if occasional {
                        scope.spawn(Self::run_occasionally(event_ids, self.trainer.model.clone()));
                    }

                    iter_count += 1;
                    if iter_count % max_iter_per_loop == 0 {
                        println!("Breaking inner loop to let threads catch up");
                        break true;
                    }
                }
            });
            if !cont {
                break;
            }
        }
        println!("Total new labels trained: {}, retrained: {}", newtrained_count, retrained_count);
        Ok(())
    }

    async fn run_occasionally(mut event_ids: Vec<EventId>, model_arc: Arc<Mutex<Option<TheModel<B>>>>) -> bool {
        if !Self::save_model_arc(model_arc) {
            return false
        }

        // TODO: check model for NaNs and concerning values?
        event_ids.sort_unstable();
        println!("EventIds being processed this time {} retrainers: {:?}", event_ids.len(), event_ids);
        true
    }

    fn save_model_arc(model_arc: Arc<Mutex<Option<TheModel<B>>>>) -> bool {
        let model_guard = match model_arc.lock() {
            Ok(model) => model,
            Err(e) => {
                println!("ERROR: Could not get lock on model to save it: {}", e);
                return false
            }
        };
        err_return!(shared_burn::model_persist::save_model(model_guard.as_ref().unwrap()), "Error saving model {}", false);
        true
    }

    fn train_event(&mut self, buf: &mut EventsWindow,
            labevents: &Vec<LabelEvent>, counts: &TrainSourceCounts)
                -> anyhow::Result<(Vec<EventId>, Vec<InputRaw>, Vec<LabelType>, Vec<ModelFloat>)> {
        let mut new_inputs = Vec::with_capacity(counts.new);
        let mut new_labels = Vec::with_capacity(counts.new);
        let mut event_timestamp_first: Timestamp = 0;
        let mut event_timestamp_last: Timestamp = 0;
        let mut last_offset_id = 0;
        for labevent in labevents {
            if let Some(data) = self.read_next_input(buf, labevent)? {
                let (new_input, new_label, event_timestamp) = data;
                new_inputs.push(new_input);
                new_labels.push(new_label);
                if event_timestamp_first == 0 {
                    event_timestamp_first = event_timestamp;
                }
                event_timestamp_last = event_timestamp;
                last_offset_id = labevent.offset_from;
            }
        }
        if new_inputs.is_empty() {
            return Ok((Vec::new(), new_inputs, Vec::new(), Vec::new()))
        }

        let retrainers = futures::executor::block_on(self.store.retrainers(counts.top, counts.oldest))?;
        let retrainer_data = RetrainerData::new(retrainers);
        let RetrainerData { event_ids, mut inputs, mut labels } = retrainer_data;
        inputs.append(&mut new_inputs);
        labels.append(&mut new_labels);

        let input: ModelInput<B> = inputs_to_device(&inputs, &self.device);
        let expected: ModelOutput<B> = outputs_to_device(labels, &self.device);
        let (outputs_tensor, losses_tensor, aggregate_loss) = self.trainer.train(input, expected)
                .map_err(|e| anyhow::anyhow!("train_batch error: {}", e))?;
        println!("Trained last offset: {}, aggloss: {}, for event times: {} -> {}", last_offset_id, aggregate_loss, to_market_datetime(event_timestamp_first), to_market_datetime(event_timestamp_last));

        let losses = tensor1_to_vec(losses_tensor);
        let outputs = tensor2_to_label_vec(outputs_tensor);

        Ok((event_ids, inputs, outputs, losses))
    }

    fn read_next_input(&mut self, buf: &mut EventsWindow, labevent: &LabelEvent) -> anyhow::Result<Option<(InputRaw, LabelType, Timestamp)>> {
        let data = buf.read_to(labevent.event_id, &self.series_events)?;
        if data.len() != SERIES1_SIZE {
            println!("Skipping labevent {}, data wrong length {} != {SERIES1_LENGTH}", labevent.offset_from, data.len());
            return Ok(None)
        }
        let last = data.back().unwrap();
        assert!(last.event_id == labevent.event_id);
        let new_input = series_to_input(data)?;
        Ok(Some((new_input, labevent.label, last.timestamp())))
    }
}

struct TrainSourceCounts {
    top: i64,
    oldest: i64,
    new: usize,
}

impl TrainSourceCounts {
    fn new(top: i64, oldest: i64, new: usize) -> Self {
        Self { top, oldest, new }
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
type LossType = BinaryCrossEntropyLossCalc<B>;
type TrainerType = Trainer<B,ModelType,LossType>;

pub fn make_trainer(device: &Device<B>, new_model: bool) -> anyhow::Result<TrainerType> {
    let (train_config, model) = if new_model {
        let model_config = TheModelConfig::default();
        shared_burn::model_persist::save_model_config(&model_config)?;
        let model = model_config.init::<TheAutodiffBackend>(device);
        // save_model(&model)?;
        let train_config = TheTrainingConfig::new(model_config, AdamWConfig::new()).with_learning_rate(1e-5);
        save_train_config(&train_config)?;
        (train_config, model)
    } else {
        (load_train_config()?, shared_burn::model_persist::load_model(device)?)
        // let model_path = artifacts_dir()?.join("model.mpk");
        // if model_path.try_exists()? {
        //     println!("Loading model... (from {:?})", &model_path);
        //     let record = CompactRecorder::new().load(model_path, &device).with_context(|| "Error loading model")?;
        //     model = config.model.init::<B>(&device).load_record(record);
        //     println!("  model loaded.")
        // }
    };

    println!("Mode: {:?}", model);
    println!("Model has {} params", model.num_params());

    let bcel = BinaryCrossEntropyLossConfig::new().with_logits(false).init(device);
    let loss_calc = BinaryCrossEntropyLossCalc(bcel);
    Trainer::new(train_config, model, loss_calc)
}
