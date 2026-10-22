//! Advisory host-page residency. Never reads or copies the borrowed input.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PageState {
    Unknown,
    Known { resident: usize, total: usize },
}

/// A racy snapshot, never an access permit. Allocation/syscall/unsupported-OS
/// failures stay unknown. Metadata is bounded to 1 MiB (at least 4 GiB input).
pub fn observe<T>(input: &[T]) -> PageState {
    if std::mem::size_of_val(input) == 0 {
        return PageState::Known {
            resident: 0,
            total: 0,
        };
    }
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    {
        // SAFETY: sysconf has no pointer arguments; a nonpositive result fails.
        let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        let Ok(page) = usize::try_from(page) else {
            return PageState::Unknown;
        };
        let Some((base, length, count)) =
            range(input.as_ptr() as usize, std::mem::size_of_val(input), page)
        else {
            return PageState::Unknown;
        };
        let mut states = Vec::<u8>::new();
        if states.try_reserve_exact(count).is_err() {
            return PageState::Unknown;
        }
        states.resize(count, 0);
        // SAFETY: the live slice borrow keeps every intersecting page mapped;
        // base is page-aligned, length covers exactly those pages, and states
        // owns count writable bytes. mincore only queries mappings, including
        // the partial first/last pages; it never dereferences input as T.
        let result = unsafe { libc::mincore(base as *mut _, length, states.as_mut_ptr().cast()) };
        if result == 0 {
            return PageState::Known {
                resident: states.iter().filter(|b| **b & 1 != 0).count(),
                total: count,
            };
        }
    }
    PageState::Unknown
}

#[cfg(any(target_os = "linux", target_os = "macos", test))]
fn range(address: usize, bytes: usize, page: usize) -> Option<(usize, usize, usize)> {
    if page == 0 || bytes == 0 {
        return None;
    }
    let base = address - address % page;
    let span = (address - base).checked_add(bytes)?;
    let count = span.checked_add(page - 1)?.checked_div(page)?;
    let length = count.checked_mul(page)?;
    base.checked_add(length)?;
    (count <= 1 << 20).then_some((base, length, count))
}

#[cfg(test)]
mod tests {
    #[test]
    fn checked_unaligned_range_and_limits() {
        assert_eq!(super::range(4097, 4096, 4096), Some((4096, 8192, 2)));
        assert_eq!(super::range(4096, 4096, 4096), Some((4096, 4096, 1)));
        for args in [
            (0, 1, 0),
            (0, 0, 4096),
            (usize::MAX, 1, 4096),
            (0, usize::MAX, 4096),
            (0, (1 << 20) + 1, 1),
        ] {
            assert_eq!(super::range(args.0, args.1, args.2), None);
        }
    }
}
