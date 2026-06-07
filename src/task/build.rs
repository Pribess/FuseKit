#![allow(dead_code)]

//! Build the Stage 3 `TaskGraph` from a Stage 2 `FusionGraph`.
//!
//! ```text
//!   FusionGraph
//!      │
//!      ▼
//!   For each Fusion:
//!     1. Pick partition strategy (largest parallel_axis, or none)
//!     2. Materialize task instance(s)
//!     3. Wire data dependencies via boundary tensor → event linking
//!      │
//!      ▼
//!   TaskGraph  (Vec<Task>, Vec<Event>, initial: Vec<TaskId>)
//! ```
//!
//! Phase 1 keeps it simple: one task per fusion, grid (1,1,1), no
//! partitioning. Partitioning across `parallel_axes` lands in Phase 2
//! once we measure where the gains are.

use anyhow::Result;
use std::collections::HashMap;

use crate::fusion::ir::FusionGraph;
use crate::graph::TensorId;
use crate::task::ir::{Event, EventId, GridAxis, PartitionPlan, Task, TaskGraph, TaskId};

pub fn build_tasks(fg: &FusionGraph) -> Result<TaskGraph> {
    let mut tasks: Vec<Task> = Vec::with_capacity(fg.fusions.len());
    let mut events: Vec<Event> = Vec::new();

    // tensor produced by fusion -> (task that produces it, completion event)
    let mut producer: HashMap<TensorId, (TaskId, EventId)> = HashMap::new();

    for fusion in &fg.fusions {
        let task_id = tasks.len() as TaskId;
        let trigger = events.len() as EventId;
        events.push(Event {
            id: trigger,
            num_triggers: 1, // one task fires this event in the trivial 1:1 mapping
            fires: Vec::new(),
        });

        // Resolve dependencies by looking up each input's source tensor
        // in the producer map. Inputs without a known producer are
        // model parameters or KV cache slots and don't need an event.
        let mut deps = Vec::new();
        for input in &fusion.inputs {
            if let Some(&(_, ev)) = producer.get(&input.source) {
                if !deps.contains(&ev) {
                    deps.push(ev);
                }
            }
        }

        // No partition in phase 1.
        let partition = PartitionPlan {
            grid: (1, 1, 1),
            bindings: Vec::new(),
        };
        let _ = GridAxis::X; // keep the enum referenced even when bindings is empty

        let task = Task {
            id: task_id,
            fusion: fusion.id,
            partition,
            input_offsets: vec![0; fusion.inputs.len()],
            output_offsets: vec![0; fusion.outputs.len()],
            deps: deps.clone(),
            trigger,
        };

        // Wire dependency targets so events know which tasks they unblock.
        for &ev in &deps {
            events[ev as usize].fires.push(task_id);
        }

        // Register this task as producer of each output boundary.
        for output in &fusion.outputs {
            producer.insert(output.source, (task_id, trigger));
        }

        tasks.push(task);
    }

    let initial: Vec<TaskId> = tasks
        .iter()
        .filter(|t| t.deps.is_empty())
        .map(|t| t.id)
        .collect();

    Ok(TaskGraph {
        tasks,
        events,
        initial,
    })
}
