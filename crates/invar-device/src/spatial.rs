//! Bounded raw spatial device ABI. Source array/fuel limits remain separate.
//! Grid construction follows cuNSearch's cell/index reordering organization,
//! using Invar's stable signed radix passes instead of a dense bounding box.
pub const MAX_POINTS: u32 = 1 << 22;
pub const LANES: u32 = 128;
pub const CUDA: &str = include_str!("spatial.cu");
/// MSL needs the private address space, supplied through `GNET_PRIVATE` as in the
/// GNET Metal preamble; the bounded arithmetic body is the CPU/CUDA text.
pub fn metal_source() -> String {
    format!(
        "#include <metal_stdlib>\n#pragma METAL fp contract(off)\nusing namespace metal;\ntypedef ulong gnet_u; typedef long gnet_w;\n#define GNET_FN inline\n#define GNET_PRIVATE thread\n{}\n{}",
        crate::EXACT_F32,
        include_str!("spatial.metal")
    )
}

pub const KERNELS: [&str; 3] = ["spatial_map", "spatial_boundaries", "spatial_query"];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    Limit,
    Length,
    InvalidGrid,
}
#[derive(Clone, Debug)]
pub struct Plan {
    parameters: [u32; 8],
    kernel: usize,
    work: u32,
    lengths: [usize; 8],
}
impl Plan {
    pub fn mapping(n: u32, cell: u32) -> Result<Self, Error> {
        if n > MAX_POINTS {
            return Err(Error::Limit);
        }
        let n = n as usize;
        Ok(Self {
            parameters: [n as u32, cell, 0, 0, 0, 0, 0, 0],
            kernel: 0,
            work: n as u32,
            lengths: [3 * n, 0, 0, 0, n, 3 * n, 0, 8],
        })
    }
    pub fn boundaries(n: u32) -> Result<Self, Error> {
        let mut p = Self::mapping(n, 0)?;
        p.kernel = 1;
        p.lengths[5] = 0;
        Ok(p)
    }
    pub fn query(
        n: u32,
        q: u32,
        cells: u32,
        stride: u32,
        radius: Option<u32>,
    ) -> Result<Self, Error> {
        if n > MAX_POINTS || q > MAX_POINTS || cells > n {
            return Err(Error::Limit);
        }
        let slots = q
            .checked_mul(stride)
            .filter(|v| *v <= (1 << 26))
            .ok_or(Error::Limit)? as usize;
        Ok(Self {
            parameters: [
                n,
                q,
                cells,
                stride,
                u32::from(radius.is_some()),
                radius.unwrap_or(0),
                0,
                0,
            ],
            kernel: 2,
            work: q,
            lengths: [
                3 * n as usize,
                3 * q as usize,
                n as usize,
                2 * cells as usize,
                q as usize,
                slots,
                slots,
                8,
            ],
        })
    }
    pub fn initial(&self, inputs: [&[u32]; 4]) -> Result<Vec<Vec<u32>>, Error> {
        if inputs.iter().zip(&self.lengths).any(|(v, n)| v.len() != *n) {
            return Err(Error::Length);
        }
        if self.kernel == 2 {
            let n = self.parameters[0];
            let mut end = 0;
            for range in inputs[3].chunks_exact(2) {
                if range[0] != end || range[1] <= end || range[1] > n {
                    return Err(Error::InvalidGrid);
                }
                end = range[1];
            }
            if end != n || inputs[0].iter().any(|v| v & 0x7fffffff >= 0x7f800000) {
                return Err(Error::InvalidGrid);
            }
            let mut ids = Vec::new();
            ids.try_reserve_exact(inputs[2].len())
                .map_err(|_| Error::Limit)?;
            ids.extend_from_slice(inputs[2]);
            ids.sort_unstable();
            if ids.last() == Some(&u32::MAX) || ids.windows(2).any(|v| v[0] == v[1]) {
                return Err(Error::InvalidGrid);
            }
        }
        let mut buffers = Vec::new();
        buffers.try_reserve_exact(8).map_err(|_| Error::Limit)?;
        for (i, &n) in self.lengths.iter().enumerate() {
            let mut v = Vec::new();
            v.try_reserve_exact(n.max(1)).map_err(|_| Error::Limit)?;
            if i < 4 {
                v.extend_from_slice(inputs[i]);
            } else if i == 7 {
                v.extend_from_slice(&self.parameters);
            }
            v.resize(n.max(1), 0);
            buffers.push(v);
        }
        Ok(buffers)
    }
    pub fn kernel(&self) -> usize {
        self.kernel
    }
    pub fn work(&self) -> u32 {
        self.work
    }
    pub fn lengths(&self) -> &[usize; 8] {
        &self.lengths
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn raw_spatial_abi_rejects_invalid_reads_and_keeps_c13_separate() {
        assert!(Plan::mapping(MAX_POINTS, 1.0f32.to_bits()).is_ok());
        assert!(Plan::mapping(MAX_POINTS + 1, 1).is_err());
        assert!(Plan::query(1, 2, 1, u32::MAX, None).is_err());
        let p = Plan::query(2, 1, 1, 3, Some(1.0f32.to_bits())).unwrap();
        let points = [0; 6];
        let queries = [0; 3];
        assert!(p.initial([&points, &queries, &[0, 1], &[0, 2]]).is_ok());
        for ranges in [[1, 2], [0, 3], [0, 0]] {
            assert_eq!(
                p.initial([&points, &queries, &[0, 1], &ranges]),
                Err(Error::InvalidGrid)
            );
        }
        for ids in [[0, 0], [0, u32::MAX]] {
            assert_eq!(
                p.initial([&points, &queries, &ids, &[0, 2]]),
                Err(Error::InvalidGrid)
            );
        }
        let mut nonfinite = points;
        nonfinite[4] = f32::NAN.to_bits();
        assert_eq!(
            p.initial([&nonfinite, &queries, &[0, 1], &[0, 2]]),
            Err(Error::InvalidGrid)
        );
        assert_eq!(
            p.initial([&points[..5], &queries, &[0, 1], &[0, 2]]),
            Err(Error::Length)
        );
    }
}
