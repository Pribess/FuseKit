// The in-binary runtime powers smoke/verification commands; production decode
// uses the embedded `python_runtime` extension. Keep the verification schema
// complete even when individual fields are not read by the CLI yet.
#![allow(dead_code)]

pub mod dispatch;
pub mod metal;
pub mod plan;
