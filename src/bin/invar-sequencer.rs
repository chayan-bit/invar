#![forbid(unsafe_code)]
fn main() {
    if let Err(error) =
        invar::system::sequencer::runtime::command(std::env::args_os().skip(1).collect())
    {
        eprintln!(
            "{{\"component\":\"sequencer\",\"success\":false,\"code\":\"S1406\",\"reason\":{}}}",
            serde_json::to_string(&error).unwrap_or_else(|_| "\"runtime failure\"".into())
        );
        std::process::exit(1);
    }
}
