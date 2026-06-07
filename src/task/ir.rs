#![allow(dead_code)]

//! Stage 3 IR — Task graph.
//!
//! A `Task` is the unit the runtime persistent megakernel dispatches.
//! One `Fusion` becomes one or more `Task`s depending on how the
//! scheduler decides to partition it across the grid.
//!
//! ```text
//!   Fusion #7 "SdpaDecode"  parallel_axes: [head, batch]
//!         │
//!         │  scheduler chooses partition: split across 24 heads
//!         ▼
//!   ┌─────────────────────────────────────────────┐
//!   │ Task #42  fusion=7  grid=(1,24,1)            │
//!   │   - 24 threadgroups, one per head            │
//!   │   - same MSL kernel function (one variant)   │
//!   │   - deps=[event_qkv_done]                    │
//!   │   - trigger=event_attn_out                   │
//!   └─────────────────────────────────────────────┘
//! ```
//!
//! Inter-task ordering rides on `Event`s. Worker threadgroups inside
//! the megakernel pull from a task queue, wait on `deps`, run the
//! kernel, then atomic-increment `trigger`'s counter — exactly the
//! MPK pattern, adapted to Metal.

use crate::fusion::ir::FusionId;

pub type TaskId = u32;
pub type EventId = u32;

/// The Stage 3 IR.
#[derive(Debug)]
pub struct TaskGraph {
    pub tasks: Vec<Task>,
    pub events: Vec<Event>,
    /// Tasks whose `deps` is empty — these are seeded into worker
    /// queues by the scheduler at megakernel launch time.
    pub initial: Vec<TaskId>,
}

#[derive(Debug, Clone)]
pub struct Task {
    pub id: TaskId,
    pub fusion: FusionId,
    pub partition: PartitionPlan,
    /// Byte offsets into each input boundary buffer for THIS task
    /// instance. When multiple tasks share a fusion (head/batch
    /// partition), each gets a different slice.
    pub input_offsets: Vec<u64>,
    pub output_offsets: Vec<u64>,
    pub deps: Vec<EventId>,
    pub trigger: EventId,
}

/// How this task's threadgroups are laid out across the dispatch grid
/// and which fusion-axis each grid dimension partitions.
///
/// `grid` follows the Metal convention: (x, y, z). `bindings` reads as
/// "grid axis X partitions the fusion's `axis_name` parallel axis".
/// An empty `bindings` slot means that grid axis does not partition
/// anything (single threadgroup along that dimension).
#[derive(Debug, Clone)]
pub struct PartitionPlan {
    pub grid: (u32, u32, u32),
    pub bindings: Vec<(GridAxis, &'static str)>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GridAxis {
    X,
    Y,
    Z,
}

/// A synchronization point in the task graph.
///
/// `num_triggers` is how many task completions must fire before
/// dependents unblock. For a fusion split across 24 head-partitioned
/// tasks, the downstream event has `num_triggers = 24`.
#[derive(Debug, Clone)]
pub struct Event {
    pub id: EventId,
    pub num_triggers: u32,
    /// Tasks that move from "waiting" to "ready" when this event
    /// completes. The scheduler enqueues them into worker queues.
    pub fires: Vec<TaskId>,
}
