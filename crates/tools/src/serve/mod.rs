//! `artemis serve`: the adapter from the engine's worker to the server's `Backend`, and the command that runs them.
pub mod backend;
pub mod panic;
pub mod run;

/// The process's exit code for what `serve` returned: 0 for a requested stop, 2 for a second device
/// loss within the window, 3 for an engine whose thread is gone. 1 is a refusal to start, which never gets here.
pub fn exit_code(r: &Result<(), server::Exit>) -> i32 {
    match r {
        Ok(()) => 0,
        Err(server::Exit::DeviceLostTwice) => 2,
        Err(server::Exit::EngineGone) => 3,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use server::Exit;

    /// A requested stop is 0, a second device loss 2, the engine's thread gone 3 (1 is a refusal to
    /// start, which never reaches `exit_code`).
    #[test]
    fn exit_codes_are_0_2_3() {
        assert_eq!(exit_code(&Ok(())), 0);
        assert_eq!(exit_code(&Err(Exit::DeviceLostTwice)), 2);
        assert_eq!(exit_code(&Err(Exit::EngineGone)), 3);
    }
}
