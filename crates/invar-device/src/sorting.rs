//! Bounded stable radix dispatches. Eight 4-bit passes use separate histogram,
//! group scan, carry scan and stable scatter kernels; no inter-group spin waits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    U32,
    I32,
    F32,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    Limit,
    Length,
}
pub const LANES: u32 = 256;
pub const KERNELS: [&str; 4] = [
    "sort_histogram",
    "sort_scan",
    "sort_carries",
    "sort_scatter",
];
pub const CUDA: &str = include_str!("sorting.cu");
pub const METAL: &str = include_str!("sorting.metal");
#[derive(Clone, Debug)]
pub struct Plan {
    length: u32,
    groups: u32,
    chunks: u32,
    kind: Kind,
}
#[derive(Clone, Debug)]
pub struct Launch {
    pub kernel: usize,
    pub blocks: u32,
    pub buffers: Vec<usize>,
}
impl Plan {
    pub fn new(length: u32, kind: Kind) -> Result<Self, Error> {
        if length > crate::MAX_DAG_ELEMENTS {
            return Err(Error::Limit);
        }
        let groups = length.div_ceil(LANES);
        Ok(Self {
            length,
            groups,
            chunks: groups.div_ceil(LANES),
            kind,
        })
    }
    pub fn length(&self) -> u32 {
        self.length
    }
    pub fn validate_inputs(&self, keys: &[u32], payload: &[u32]) -> Result<(), Error> {
        if keys.len() != self.length as usize || payload.len() != keys.len() {
            Err(Error::Length)
        } else {
            Ok(())
        }
    }
    pub fn parameters(&self, pass: usize) -> [u32; 5] {
        [
            self.length,
            self.groups,
            self.chunks,
            match self.kind {
                Kind::U32 => 0,
                Kind::I32 => 1,
                Kind::F32 => 2,
            },
            (pass.min(7) * 4) as u32,
        ]
    }
    pub fn buffer_words(&self) -> [usize; 16] {
        let n = self.length as usize;
        let h = self.groups as usize * 16;
        let c = self.chunks as usize * 16;
        [n, n, n, n, h, h, c, c, 5, 5, 5, 5, 5, 5, 5, 5]
    }
    pub fn launches(&self) -> Vec<Launch> {
        if self.length == 0 {
            return Vec::new();
        }
        let mut result = Vec::with_capacity(32);
        for pass in 0..8 {
            let (keys, values, out_keys, out_values) = if pass % 2 == 0 {
                (0, 1, 2, 3)
            } else {
                (2, 3, 0, 1)
            };
            let p = 8 + pass;
            result.extend([
                Launch {
                    kernel: 0,
                    blocks: self.groups,
                    buffers: vec![keys, 4, p],
                },
                Launch {
                    kernel: 1,
                    blocks: 16 * self.chunks,
                    buffers: vec![4, 5, 6, p],
                },
                Launch {
                    kernel: 2,
                    blocks: 1,
                    buffers: vec![6, 7, p],
                },
                Launch {
                    kernel: 3,
                    blocks: self.groups,
                    buffers: vec![keys, values, 5, 7, out_keys, out_values, p],
                },
            ]);
        }
        result
    }
}
