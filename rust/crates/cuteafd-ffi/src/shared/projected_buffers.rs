use crate::CuteafdDeviceBuffer as Buffer;

pub(super) fn validate_buffers<const N: usize>(
    items: &[(Buffer, usize); N],
    alignments: &[usize; N],
    writable: &[usize],
) -> Result<(), &'static str> {
    let device = items[0].0.device_id;
    if device < 0
        || items.iter().zip(alignments).any(|((b, n), alignment)| {
            b.ptr.is_null()
                || b.bytes < *n
                || b.flags != 0
                || b.device_id != device
                || b.ptr as usize % alignment != 0
                || *n > isize::MAX as usize
                || (b.ptr as usize).checked_add(*n).is_none()
        })
    {
        return Err("buffer extent/alignment/device");
    }
    for &i in writable {
        let (a, na) = items[i];
        if items.iter().enumerate().any(|(j, (b, nb))| {
            i != j
                && (a.ptr as usize) < b.ptr as usize + nb
                && (b.ptr as usize) < a.ptr as usize + na
        }) {
            return Err("overlapping writable operands");
        }
    }
    Ok(())
}

#[cfg(test)]
pub(super) fn assert_operand_contract<S>(
    program: &str,
    names: &[&str],
    operands: impl IntoIterator<Item = (&'static str, S, &'static str)>,
    alignments: &[usize],
) {
    let contracts: serde_json::Value = serde_json::from_str(include_str!(
        "../../tests/fixtures/projected_operand_contracts.json"
    ))
    .unwrap();
    let operands: Vec<_> = operands.into_iter().collect();
    assert_eq!(names.len(), operands.len());
    assert_eq!(names.len(), alignments.len());
    let actual: Vec<_> = names
        .iter()
        .zip(operands)
        .zip(alignments)
        .map(|((name, (dtype, _, _)), align)| {
            serde_json::json!({"name": name, "dtype": dtype, "align": align})
        })
        .collect();
    assert_eq!(serde_json::json!(actual), contracts[program], "{program}");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_extents_beyond_pointer_offset_limit() {
        let buffer = Buffer {
            ptr: 16usize as *mut std::ffi::c_void,
            bytes: usize::MAX,
            device_id: 0,
            flags: 0,
        };
        assert!(validate_buffers(&[(buffer, isize::MAX as usize + 1)], &[16], &[]).is_err());
    }
}
