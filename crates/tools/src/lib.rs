//! The library behind the `artemis` binary: tap sets on disk, per-tap comparison, the pass rules of test 2, the GPU's
//! taps and tests 3 and 5, generate's loop and record.
pub mod compare;
pub mod generate;
pub mod hits;
pub mod kl;
pub mod lookup_sim;
pub mod mtp_check;
pub mod profile;
pub mod serve;
pub mod setup;
pub mod snaptimes;
pub mod spec_check;
pub mod speed;
pub mod tapdir;
pub mod test2;
pub mod test3;
pub mod test5;
pub mod test6;
pub mod test7;
pub mod verify_cost;
