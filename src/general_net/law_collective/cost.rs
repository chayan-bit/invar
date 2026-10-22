//! Placement cost model: refuse a P8 reduction to the native path when even
//! an ideal collective cannot beat it. Every figure comes from a recorded
//! per-device calibration; nothing here is a baked-in device constant.

use super::{Error, LawRepr};

/// Measured throughputs of one device and one law, in bytes per millisecond.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Figures {
    /// Device memory bandwidth, from a device-to-device copy probe.
    pub peak_bytes_per_ms: f64,
    /// Staging of host-resident input, as the warm path does it.
    pub link_bytes_per_ms: f64,
    /// The native path's fold of this law over the same input words.
    pub native_bytes_per_ms: f64,
}

/// A recorded calibration: several independent passes, the best (fastest)
/// value of each figure, which decides the route, and the worst (slowest),
/// which breaks a marginal decision.
#[derive(Clone, Debug, PartialEq)]
pub struct Calibration {
    pub best: Figures,
    pub worst: Figures,
    /// Which probe produced these figures, on which device, and when.
    pub source: String,
}

/// Where the reduction's input lives when the call starts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Residency {
    Device,
    Host,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Route {
    Collective,
    Native(&'static str),
}

/// A route with the evidence behind it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Decision {
    pub route: Route,
    /// Collective floor over native time from the best figures, when compared.
    pub ratio: Option<f64>,
    /// The ratio is within `MARGIN` of `TIE`, so the worst figures decided.
    pub marginal: bool,
}

/// A refusal needs the collective's ideal time to reach this share of native.
pub const TIE: f64 = 0.95;
/// Relative band around `TIE` inside which a decision is marginal.
pub const MARGIN: f64 = 0.05;

const KEYS: [&str; 3] = [
    "peak_bytes_per_ms",
    "link_bytes_per_ms",
    "native_bytes_per_ms",
];

fn figures(value: &serde_json::Value) -> Result<Figures, Error> {
    let [peak, link, native] = KEYS.map(|key| {
        value[key]
            .as_f64()
            .filter(|x| x.is_finite() && *x > 0.0)
            .ok_or(Error::Unsupported("calibration figure missing or invalid"))
    });
    Ok(Figures {
        peak_bytes_per_ms: peak?,
        link_bytes_per_ms: link?,
        native_bytes_per_ms: native?,
    })
}

impl Figures {
    fn get(&self, key: &str) -> f64 {
        match key {
            "peak_bytes_per_ms" => self.peak_bytes_per_ms,
            "link_bytes_per_ms" => self.link_bytes_per_ms,
            _ => self.native_bytes_per_ms,
        }
    }
}

impl Calibration {
    /// Parse one recorded calibration. Its best and worst figures must be
    /// exactly the maximum and minimum over its recorded passes.
    pub fn from_json(value: &serde_json::Value) -> Result<Self, Error> {
        let source = value["source"]
            .as_str()
            .filter(|s| !s.trim().is_empty())
            .ok_or(Error::Unsupported("calibration has no recorded source"))?;
        let passes = value["passes"]
            .as_array()
            .filter(|p| !p.is_empty())
            .ok_or(Error::Unsupported("calibration has no recorded passes"))?
            .iter()
            .map(figures)
            .collect::<Result<Vec<_>, _>>()?;
        let (best, worst) = (figures(value)?, figures(&value["worst"])?);
        for key in KEYS {
            let all = passes.iter().map(|p| p.get(key));
            let max = all.clone().fold(f64::MIN, f64::max);
            let min = all.fold(f64::MAX, f64::min);
            if best.get(key) != max || worst.get(key) != min {
                return Err(Error::Unsupported(
                    "calibration best/worst do not match its passes",
                ));
            }
        }
        Ok(Self {
            best,
            worst,
            source: source.to_owned(),
        })
    }
}

/// Scans stay collective. A device-resident reduction stays collective: the
/// native path would first read every byte back, so it cannot beat the
/// collective's floor. A host-resident reduction is refused when the
/// collective's lower bound (staging at link speed plus one read at peak, no
/// launch cost) is at least `TIE` of the native fold time.
pub fn route(
    repr: &LawRepr,
    scan: bool,
    length: u32,
    residency: Residency,
    calibration: &Calibration,
) -> Route {
    decide(repr, scan, length, residency, calibration).route
}

/// `route` with its ratio and whether it was marginal. Within `MARGIN` of
/// `TIE`, the path whose slowest calibration pass is faster is taken, so a
/// given file always gives the same answer and a busy pass cannot flip it.
pub fn decide(
    repr: &LawRepr,
    scan: bool,
    length: u32,
    residency: Residency,
    calibration: &Calibration,
) -> Decision {
    decide_with(repr, scan, length, residency, calibration, Mutation::None)
}

#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mutation {
    None,
    DropHostLink,
    RefuseDeviceResident,
    RefuseScan,
    IgnoreMargin,
}

#[doc(hidden)]
pub fn route_with(
    repr: &LawRepr,
    scan: bool,
    length: u32,
    residency: Residency,
    calibration: &Calibration,
    mutation: Mutation,
) -> Route {
    decide_with(repr, scan, length, residency, calibration, mutation).route
}

#[doc(hidden)]
pub fn decide_with(
    repr: &LawRepr,
    scan: bool,
    length: u32,
    residency: Residency,
    calibration: &Calibration,
    mutation: Mutation,
) -> Decision {
    let collective = Decision {
        route: Route::Collective,
        ratio: None,
        marginal: false,
    };
    if scan && mutation != Mutation::RefuseScan {
        return collective;
    }
    let host = match residency {
        Residency::Host => mutation != Mutation::DropHostLink,
        Residency::Device if mutation == Mutation::RefuseDeviceResident => true,
        Residency::Device => return collective,
    };
    let bytes = f64::from(length) * f64::from(repr.word_count()) * 4.0;
    if bytes == 0.0 {
        return collective;
    }
    let times = |f: &Figures| {
        let link = if host {
            bytes / f.link_bytes_per_ms
        } else {
            0.0
        };
        (
            link + bytes / f.peak_bytes_per_ms,
            bytes / f.native_bytes_per_ms,
        )
    };
    let (floor, native) = times(&calibration.best);
    let ratio = floor / native;
    let marginal = (ratio - TIE).abs() <= MARGIN * TIE && mutation != Mutation::IgnoreMargin;
    let refuse = if marginal {
        let (floor, native) = times(&calibration.worst);
        native < floor
    } else {
        ratio >= TIE
    };
    Decision {
        route: if refuse {
            Route::Native("P8 reduction cannot beat the native fold on host-resident input")
        } else {
            Route::Collective
        },
        ratio: Some(ratio),
        marginal,
    }
}
