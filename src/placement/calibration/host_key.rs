//! A stable per-machine key for calibration receipts, read without privileges.
#[cfg(target_os = "linux")]
pub(super) fn read() -> Option<String> {
    std::fs::read_to_string("/etc/machine-id").ok()
}

#[cfg(target_os = "macos")]
pub(super) fn read() -> Option<String> {
    super::command("ioreg", &["-rd1", "-c", "IOPlatformExpertDevice"]).and_then(|s| {
        s.lines()
            .find(|l| l.contains("\"IOPlatformUUID\""))
            .and_then(|l| l.split_once('=').map(|(_, v)| v.trim().to_owned()))
    })
}

/// `reg query` prints `MachineGuid <type> <guid>`.
#[cfg(windows)]
pub(super) fn read() -> Option<String> {
    let key = r"HKLM\SOFTWARE\Microsoft\Cryptography";
    super::command("reg", &["query", key, "/v", "MachineGuid"]).and_then(|s| {
        s.lines()
            .find_map(|l| match l.split_whitespace().collect::<Vec<_>>()[..] {
                ["MachineGuid", _, guid] => Some(guid.to_owned()),
                _ => None,
            })
    })
}

#[cfg(not(any(target_os = "linux", target_os = "macos", windows)))]
pub(super) fn read() -> Option<String> {
    None
}
