//! A warning for an API key file that other users can read. A warning only: the server still starts, and a file whose
//! permissions cannot be read gives no warning.

use std::path::Path;

/// The warning text when `path` is readable by users other than its owner, else `None`. Unix: any group or other bit
/// of the mode (of the target, for a symlink); the text names what the mode grants. Windows: an allow rule granting
/// read (FILE_READ_DATA, GENERIC_READ or GENERIC_ALL) to Everyone, Users, Authenticated Users, Guests, Interactive or
/// Anonymous. A file whose permissions cannot be read gives `None`.
pub fn exposure(path: &Path) -> Option<String> {
    let name = path.display();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        // `metadata` follows a symlink: the target's mode is what decides who can read the key.
        let mode = std::fs::metadata(path).ok()?.permissions().mode();
        unix_warning(&name.to_string(), mode)
    }
    #[cfg(not(unix))]
    {
        exposed(path).then(|| {
            format!(
                "api_key_file {name} can be read by other users; restrict it with: {}",
                windows_remedy(&name.to_string())
            )
        })
    }
}

/// The warning for a key file at `name` with `mode`, or `None` when only its owner has access.
#[cfg(any(unix, test))]
fn unix_warning(name: &str, mode: u32) -> Option<String> {
    unix_grant(mode).map(|grant| {
        format!(
            "api_key_file {name} grants access to {grant}; restrict it with: chmod 600 {}",
            shell_quote(name)
        )
    })
}

/// What the group and other bits of `mode` grant, as "group (read, write) and others (read)", or `None` when both are
/// empty.
#[cfg(any(unix, test))]
fn unix_grant(mode: u32) -> Option<String> {
    let part = |who: &str, bits: u32| {
        let names: Vec<&str> = [(4, "read"), (2, "write"), (1, "execute")]
            .into_iter()
            .filter(|(b, _)| bits & b != 0)
            .map(|(_, n)| n)
            .collect();
        (!names.is_empty()).then(|| format!("{who} ({})", names.join(", ")))
    };
    let parts: Vec<String> = [part("group", (mode >> 3) & 7), part("others", mode & 7)]
        .into_iter()
        .flatten()
        .collect();
    (!parts.is_empty()).then(|| parts.join(" and "))
}

/// `path` single-quoted for a POSIX shell (a single quote in it becomes `'\''`).
#[cfg(any(unix, test))]
fn shell_quote(path: &str) -> String {
    format!("'{}'", path.replace('\'', r"'\''"))
}

/// The PowerShell command that restricts the file to its owner. The path is single-quoted (a `$` or a backtick in it
/// stays literal, a single quote is doubled); `${env:USERNAME}` stays in double quotes so it expands.
#[cfg(any(not(unix), test))]
fn windows_remedy(path: &str) -> String {
    let quoted = path.replace('\'', "''");
    format!("icacls '{quoted}' /inheritance:r /grant:r \"${{env:USERNAME}}:(R,W)\"")
}

/// Whether an ACE's access mask grants reading the file's data: FILE_READ_DATA, or the generic read or all bits, which
/// NTFS maps on write but another file system may keep unmapped. An allow and a deny ACE are judged alike.
#[cfg(any(windows, test))]
fn grants_read(mask: u32) -> bool {
    const FILE_READ_DATA: u32 = 0x0000_0001;
    const GENERIC_ALL: u32 = 0x1000_0000;
    const GENERIC_READ: u32 = 0x8000_0000;
    mask & (FILE_READ_DATA | GENERIC_READ | GENERIC_ALL) != 0
}

#[cfg(windows)]
fn exposed(path: &Path) -> bool {
    use std::os::windows::ffi::OsStrExt;
    use std::ptr::null_mut;
    use windows_sys::Win32::Foundation::{LocalFree, ERROR_SUCCESS};
    use windows_sys::Win32::Security::Authorization::{
        ConvertStringSidToSidW, GetNamedSecurityInfoW, SE_FILE_OBJECT,
    };
    use windows_sys::Win32::Security::{
        AclSizeInformation, EqualSid, GetAce, GetAclInformation, ACCESS_ALLOWED_ACE,
        ACL_SIZE_INFORMATION, DACL_SECURITY_INFORMATION,
    };

    const ACCESS_ALLOWED: u8 = 0; // ACCESS_ALLOWED_ACE_TYPE
    const ACCESS_DENIED: u8 = 1; // ACCESS_DENIED_ACE_TYPE
    const INHERIT_ONLY: u8 = 0x08; // INHERIT_ONLY_ACE
    let wide: Vec<u16> = path.as_os_str().encode_wide().chain([0]).collect();
    let (mut dacl, mut sd) = (null_mut(), null_mut());
    // SAFETY: every pointer is valid for its call; the descriptor (which owns the DACL) and each converted SID are
    // freed once, below.
    unsafe {
        let r = GetNamedSecurityInfoW(
            wide.as_ptr(),
            SE_FILE_OBJECT,
            DACL_SECURITY_INFORMATION,
            null_mut(),
            null_mut(),
            &mut dacl,
            null_mut(),
            &mut sd,
        );
        if r != ERROR_SUCCESS {
            return false;
        }
        let found = if dacl.is_null() {
            true // no DACL: everyone has full access
        } else {
            let mut info: ACL_SIZE_INFORMATION = std::mem::zeroed();
            let ok = GetAclInformation(
                dacl,
                &mut info as *mut _ as *mut _,
                std::mem::size_of::<ACL_SIZE_INFORMATION>() as u32,
                AclSizeInformation,
            );
            let broad: Vec<_> = [
                "S-1-1-0",      // Everyone
                "S-1-5-32-545", // Users
                "S-1-5-11",     // Authenticated Users
                "S-1-5-32-546", // Guests
                "S-1-5-4",      // Interactive
                "S-1-5-7",      // Anonymous
            ]
            .iter()
            .filter_map(|s| {
                let w: Vec<u16> = s.encode_utf16().chain([0]).collect();
                let mut sid = null_mut();
                (ConvertStringSidToSidW(w.as_ptr(), &mut sid) != 0).then_some(sid)
            })
            .collect();
            let mut hit = false;
            let mut denied = vec![false; broad.len()];
            if ok != 0 {
                for i in 0..info.AceCount {
                    let mut ace = null_mut();
                    if GetAce(dacl, i, &mut ace) == 0 {
                        break;
                    }
                    // An allowed and a denied ACE share this layout; walk in order, as access checks do.
                    let a = ace as *const ACCESS_ALLOWED_ACE;
                    let kind = (*a).Header.AceType;
                    if (kind != ACCESS_ALLOWED && kind != ACCESS_DENIED)
                        || (*a).Header.AceFlags & INHERIT_ONLY != 0
                        || !grants_read((*a).Mask)
                    {
                        continue;
                    }
                    let sid = std::ptr::addr_of!((*a).SidStart) as *mut _;
                    let Some(n) = broad.iter().position(|b| EqualSid(sid, *b) != 0) else {
                        continue;
                    };
                    if kind == ACCESS_DENIED {
                        denied[n] = true;
                    } else if !denied[n] {
                        hit = true;
                        break;
                    }
                }
            }
            for b in broad {
                LocalFree(b);
            }
            hit
        };
        LocalFree(sd);
        found
    }
}

#[cfg(not(any(unix, windows)))]
fn exposed(_: &Path) -> bool {
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(tag: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("artemis-keyfile-{}-{tag}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[cfg(unix)]
    fn file_with_mode(dir: &Path, name: &str, mode: u32) -> std::path::PathBuf {
        use std::os::unix::fs::PermissionsExt;
        let p = dir.join(name);
        std::fs::write(&p, b"key\n").unwrap();
        std::fs::set_permissions(&p, std::fs::Permissions::from_mode(mode)).unwrap();
        p
    }

    #[cfg(unix)]
    #[test]
    fn keyfile_0600_gives_none() {
        let d = scratch("u600");
        assert_eq!(exposure(&file_with_mode(&d, "k", 0o600)), None);
        let _ = std::fs::remove_dir_all(d);
    }

    #[cfg(unix)]
    #[test]
    fn keyfile_0640_and_0604_warn_naming_the_file_and_chmod() {
        let d = scratch("uwarn");
        for mode in [0o640, 0o604] {
            let p = file_with_mode(&d, &format!("k{mode:o}"), mode);
            let w = exposure(&p).unwrap_or_else(|| panic!("no warning for {mode:o}"));
            assert!(
                w.contains(&p.display().to_string())
                    && w.contains(" grants access to ")
                    && w.contains("chmod 600"),
                "{w}"
            );
        }
        let _ = std::fs::remove_dir_all(d);
    }

    #[cfg(unix)]
    #[test]
    fn keyfile_symlink_uses_the_targets_mode() {
        let d = scratch("usym");
        let open = file_with_mode(&d, "open", 0o644);
        let link = d.join("link");
        std::os::unix::fs::symlink(&open, &link).unwrap();
        assert!(exposure(&link).is_some());
        let shut = file_with_mode(&d, "shut", 0o600);
        let link2 = d.join("link2");
        std::os::unix::fs::symlink(&shut, &link2).unwrap();
        assert_eq!(exposure(&link2), None);
        let _ = std::fs::remove_dir_all(d);
    }

    #[test]
    fn the_windows_remedy_single_quotes_the_path() {
        // A `$` or a backtick in a path must not expand; a single quote is doubled; the user name still expands.
        let r = windows_remedy("C:/k/$x`y'z");
        assert_eq!(
            r,
            "icacls 'C:/k/$x`y''z' /inheritance:r /grant:r \"${env:USERNAME}:(R,W)\""
        );
    }

    #[test]
    fn the_unix_remedy_single_quotes_the_path() {
        assert_eq!(shell_quote("/k/a b$c'd"), r"'/k/a b$c'\''d'");
    }

    #[test]
    fn the_unix_grant_names_what_the_mode_gives() {
        assert_eq!(unix_grant(0o600), None);
        assert_eq!(unix_grant(0o640).as_deref(), Some("group (read)"));
        assert_eq!(unix_grant(0o620).as_deref(), Some("group (write)"));
        assert_eq!(unix_grant(0o604).as_deref(), Some("others (read)"));
        assert_eq!(
            unix_grant(0o666).as_deref(),
            Some("group (read, write) and others (read, write)")
        );
        assert_eq!(unix_grant(0o610).as_deref(), Some("group (execute)"));
    }

    #[test]
    fn the_unix_warning_names_the_file_the_grant_and_the_remedy() {
        assert_eq!(unix_warning("/k/key", 0o600), None);
        assert_eq!(
            unix_warning("/k/key", 0o644).as_deref(),
            Some(
                "api_key_file /k/key grants access to group (read) and others (read); restrict it with: \
                 chmod 600 '/k/key'"
            )
        );
        assert_eq!(
            unix_warning("/k/a b", 0o604).as_deref(),
            Some("api_key_file /k/a b grants access to others (read); restrict it with: chmod 600 '/k/a b'")
        );
    }

    #[test]
    fn the_read_mask_counts_generic_read_and_all() {
        // An ACE can hold the generic bits unmapped (a file system that does not map them); they grant read.
        assert!(grants_read(0x0001)); // FILE_READ_DATA
        assert!(grants_read(0x8000_0000)); // GENERIC_READ
        assert!(grants_read(0x1000_0000)); // GENERIC_ALL
        assert!(grants_read(0x0012_0089)); // FR
        assert!(!grants_read(0x4000_0000)); // GENERIC_WRITE
        assert!(!grants_read(0x2000_0000)); // GENERIC_EXECUTE
        assert!(!grants_read(0x0010_0000 | 0x0002_0000)); // SYNCHRONIZE | READ_CONTROL
    }

    #[test]
    fn keyfile_missing_file_gives_none() {
        assert_eq!(exposure(Path::new("no-such-artemis-key-file")), None);
    }

    #[cfg(windows)]
    fn icacls(args: &[&str]) {
        let s = std::process::Command::new("icacls")
            .args(args)
            .output()
            .unwrap();
        assert!(s.status.success(), "{}", String::from_utf8_lossy(&s.stdout));
    }

    #[cfg(windows)]
    #[test]
    fn keyfile_owner_only_acl_gives_none_and_everyone_warns() {
        let d = scratch("win");
        let p = d.join("k");
        std::fs::write(&p, b"key\n").unwrap();
        let f = p.to_str().unwrap();
        let me = format!("{}:(R,W)", std::env::var("USERNAME").unwrap());
        icacls(&[f, "/inheritance:r", "/grant:r", &me]);
        assert_eq!(exposure(&p), None);
        icacls(&[f, "/grant", "*S-1-1-0:R"]);
        let w = exposure(&p).expect("Everyone can read");
        assert!(w.contains(&format!("icacls '{f}' ")), "{w}");
        assert!(w.contains("\"${env:USERNAME}:(R,W)\""), "{w}");
        let _ = std::fs::remove_dir_all(d);
    }

    /// A fresh file only its owner can read.
    #[cfg(windows)]
    fn owner_only(tag: &str) -> (std::path::PathBuf, std::path::PathBuf) {
        let d = scratch(tag);
        let p = d.join("k");
        std::fs::write(&p, b"key\n").unwrap();
        let me = format!("{}:(R,W)", std::env::var("USERNAME").unwrap());
        icacls(&[p.to_str().unwrap(), "/inheritance:r", "/grant:r", &me]);
        (d, p)
    }

    /// Adds ReadData rules, `(SID, "Allow" or "Deny")`, in order, with .NET's FileSecurity in Windows PowerShell:
    /// icacls drops a deny when the same account is granted afterwards, and cannot order rules.
    #[cfg(windows)]
    fn add_rules(f: &str, rules: &[(&str, &str)]) {
        let mut script = format!("$f = '{f}'; $a = [IO.File]::GetAccessControl($f); ");
        for (sid, kind) in rules {
            script.push_str(&format!(
                "$a.AddAccessRule((New-Object Security.AccessControl.FileSystemAccessRule(\
                 [Security.Principal.SecurityIdentifier]'{sid}', 'ReadData', '{kind}'))); "
            ));
        }
        script.push_str("[IO.File]::SetAccessControl($f, $a)");
        let out = std::process::Command::new("powershell")
            .args(["-NoProfile", "-Command", &script])
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
    }

    #[cfg(windows)]
    #[test]
    fn keyfile_a_deny_before_an_allow_for_everyone_gives_none() {
        let (d, p) = owner_only("windeny");
        add_rules(
            p.to_str().unwrap(),
            &[("S-1-1-0", "Deny"), ("S-1-1-0", "Allow")],
        );
        assert_eq!(exposure(&p), None);
        let _ = std::fs::remove_dir_all(d);
    }

    #[cfg(windows)]
    #[test]
    fn keyfile_a_deny_for_users_does_not_hide_an_allow_for_everyone() {
        let (d, p) = owner_only("windenyusers");
        add_rules(
            p.to_str().unwrap(),
            &[("S-1-5-32-545", "Deny"), ("S-1-1-0", "Allow")],
        );
        assert!(exposure(&p).is_some(), "Everyone still counts");
        let _ = std::fs::remove_dir_all(d);
    }

    /// NTFS keeps no inherit-only rule on a file, so the rule goes on a directory (`exposure` reads a directory's DACL
    /// the same way): Everyone gets read through an `(OI)(IO)` rule, which is for the children and grants nothing
    /// here.
    #[cfg(windows)]
    #[test]
    fn keyfile_an_inherit_only_allow_is_skipped() {
        let d = scratch("wininh");
        let f = d.to_str().unwrap();
        let me = format!("{}:(OI)(CI)(F)", std::env::var("USERNAME").unwrap());
        icacls(&[f, "/inheritance:r", "/grant:r", &me]);
        assert_eq!(exposure(&d), None);
        icacls(&[f, "/grant", "*S-1-1-0:(OI)(IO)R"]);
        // The rule must be in the DACL for this test to mean anything.
        let out = std::process::Command::new("icacls")
            .arg(f)
            .output()
            .unwrap();
        let listing = String::from_utf8_lossy(&out.stdout);
        assert!(listing.contains("Everyone:(OI)(IO)"), "{listing}");
        assert_eq!(exposure(&d), None, "an inherit-only allow grants nothing");
        // Without the inherit-only flag the same rule counts.
        icacls(&[f, "/grant", "*S-1-1-0:(OI)R"]);
        assert!(exposure(&d).is_some());
        let _ = std::fs::remove_dir_all(d);
    }

    /// `icacls (GR)` and `(GA)` grant read as `R` does. NTFS stores them mapped (as FR and FA), so this covers the
    /// whole path; `the_read_mask_counts_generic_read_and_all` covers the unmapped bits.
    #[cfg(windows)]
    #[test]
    fn keyfile_generic_read_or_all_for_everyone_warns() {
        for (tag, right) in [("wingr", "(GR)"), ("winga", "(GA)")] {
            let (d, p) = owner_only(tag);
            icacls(&[p.to_str().unwrap(), "/grant", &format!("*S-1-1-0:{right}")]);
            assert!(exposure(&p).is_some(), "{right}");
            let _ = std::fs::remove_dir_all(d);
        }
    }

    /// A deny asked for as generic read still comes before the allow and wins.
    #[cfg(windows)]
    #[test]
    fn keyfile_a_generic_read_deny_before_an_allow_for_everyone_gives_none() {
        let (d, p) = owner_only("windenygr");
        let f = p.to_str().unwrap();
        icacls(&[f, "/deny", "*S-1-1-0:(GR)"]);
        add_rules(f, &[("S-1-1-0", "Allow")]);
        assert_eq!(exposure(&p), None);
        let _ = std::fs::remove_dir_all(d);
    }

    #[cfg(windows)]
    #[test]
    fn keyfile_guests_interactive_and_anonymous_count_as_other_users() {
        for (tag, sid) in [
            ("winguests", "S-1-5-32-546"),
            ("wininter", "S-1-5-4"),
            ("winanon", "S-1-5-7"),
        ] {
            let (d, p) = owner_only(tag);
            icacls(&[p.to_str().unwrap(), "/grant", &format!("*{sid}:R")]);
            assert!(exposure(&p).is_some(), "{sid}");
            let _ = std::fs::remove_dir_all(d);
        }
    }
}
