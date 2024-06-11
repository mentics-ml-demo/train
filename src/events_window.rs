use std::{cmp::Ordering, collections::VecDeque};
use anyhow::bail;

use quote::QuoteEvent;
use shared_types::*;
use series_store::SeriesReader;

pub struct EventsWindow {
    target_length: usize,
    events: VecDeque<QuoteEvent>,
}

impl EventsWindow { // <'a>
    pub fn new(target_length: usize) -> Self {
        Self { target_length, events: VecDeque::new() }
    }

    pub fn read_to(&mut self, event_id: EventId, series_events: &SeriesReader) -> anyhow::Result<&VecDeque<QuoteEvent>> {
        // let mut count = 0;
        loop {
            let event: QuoteEvent = series_events.read_into_event()?;
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
