//! Interval types for Goth
//!
//! Track value ranges at the type level: F64⊢[0..1]

use serde::{Deserialize, Serialize};

/// A bound (endpoint) of an interval
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Bound {
    /// Negative infinity
    NegInf,

    /// Positive infinity
    PosInf,

    /// Concrete value
    Const(f64),

    /// Symbolic bound (type variable or expression)
    Var(Box<str>),
}

/// Whether a bound is inclusive or exclusive
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BoundKind {
    Inclusive, // [ or ]
    Exclusive, // ( or )
}

/// An interval [a..b], (a..b), [a..b), (a..b]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Interval {
    pub lo: Bound,
    pub lo_kind: BoundKind,
    pub hi: Bound,
    pub hi_kind: BoundKind,
}

/// Interval set (union of intervals)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IntervalSet(pub Vec<Interval>);

impl Interval {
    /// Closed interval [lo..hi]
    pub fn closed(lo: Bound, hi: Bound) -> Self {
        Interval {
            lo,
            lo_kind: BoundKind::Inclusive,
            hi,
            hi_kind: BoundKind::Inclusive,
        }
    }

    /// Open interval (lo..hi)
    pub fn open(lo: Bound, hi: Bound) -> Self {
        Interval {
            lo,
            lo_kind: BoundKind::Exclusive,
            hi,
            hi_kind: BoundKind::Exclusive,
        }
    }

    /// Half-open [lo..hi)
    pub fn half_open_right(lo: Bound, hi: Bound) -> Self {
        Interval {
            lo,
            lo_kind: BoundKind::Inclusive,
            hi,
            hi_kind: BoundKind::Exclusive,
        }
    }

    /// Half-open (lo..hi]
    pub fn half_open_left(lo: Bound, hi: Bound) -> Self {
        Interval {
            lo,
            lo_kind: BoundKind::Exclusive,
            hi,
            hi_kind: BoundKind::Inclusive,
        }
    }

    /// Unit interval [0..1]
    pub fn unit() -> Self {
        Interval::closed(Bound::Const(0.0), Bound::Const(1.0))
    }

    /// Non-negative [0..∞)
    pub fn non_negative() -> Self {
        Interval::half_open_right(Bound::Const(0.0), Bound::PosInf)
    }

    /// Positive (0..∞)
    pub fn positive() -> Self {
        Interval::open(Bound::Const(0.0), Bound::PosInf)
    }

    /// All reals (-∞..∞)
    pub fn all() -> Self {
        Interval::open(Bound::NegInf, Bound::PosInf)
    }

    /// Check if this interval might contain zero
    pub fn may_contain_zero(&self) -> bool {
        // Check if 0 is >= lower bound (respecting inclusivity)
        let above_lo = match &self.lo {
            Bound::NegInf => true,
            Bound::PosInf => false,
            Bound::Const(lo) => {
                if *lo < 0.0 {
                    true
                } else if *lo == 0.0 {
                    self.lo_kind == BoundKind::Inclusive
                } else {
                    false
                }
            }
            Bound::Var(_) => true, // Conservative: symbolic bounds might allow zero
        };
        
        // Check if 0 is <= upper bound (respecting inclusivity)
        let below_hi = match &self.hi {
            Bound::NegInf => false,
            Bound::PosInf => true,
            Bound::Const(hi) => {
                if *hi > 0.0 {
                    true
                } else if *hi == 0.0 {
                    self.hi_kind == BoundKind::Inclusive
                } else {
                    false
                }
            }
            Bound::Var(_) => true, // Conservative: symbolic bounds might allow zero
        };
        
        above_lo && below_hi
    }
}

impl Bound {
    pub fn is_finite(&self) -> bool {
        matches!(self, Bound::Const(_) | Bound::Var(_))
    }
}

impl IntervalSet {
    pub fn single(interval: Interval) -> Self {
        IntervalSet(vec![interval])
    }

    pub fn union(mut self, other: Interval) -> Self {
        self.0.push(other);
        // TODO: normalize/merge overlapping intervals
        self
    }

    /// The "tainted" interval (undefined, e.g., from 0-division)
    pub fn undefined() -> Self {
        IntervalSet(vec![])
    }

    pub fn is_undefined(&self) -> bool {
        self.0.is_empty()
    }
}

impl std::fmt::Display for Bound {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Bound::NegInf => write!(f, "-∞"),
            Bound::PosInf => write!(f, "∞"),
            Bound::Const(v) => write!(f, "{}", v),
            Bound::Var(name) => write!(f, "{}", name),
        }
    }
}

impl std::fmt::Display for Interval {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let lo_bracket = match self.lo_kind {
            BoundKind::Inclusive => "[",
            BoundKind::Exclusive => "(",
        };
        let hi_bracket = match self.hi_kind {
            BoundKind::Inclusive => "]",
            BoundKind::Exclusive => ")",
        };
        write!(f, "{}{}..{}{}", lo_bracket, self.lo, self.hi, hi_bracket)
    }
}

impl std::fmt::Display for IntervalSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_undefined() {
            write!(f, "⊥")
        } else {
            let parts: Vec<_> = self.0.iter().map(|i| format!("{}", i)).collect();
            write!(f, "{}", parts.join(" ∪ "))
        }
    }
}
