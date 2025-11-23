// lib.rs - Rust module for high-performance FVG and S/R detection
use pyo3::prelude::*;

#[derive(Clone, Debug)]
pub struct FVGData {
    pub fvg_type: String,  // "bullish" or "bearish"
    pub level1: f64,
    pub level2: f64,
    pub index: usize,
}

/// Detects Fair Value Gaps (FVGs) in historical price data
/// Returns a vector where each element is either None or a tuple of (type, level1, level2, index)
#[pyfunction]
pub fn detect_fvg_rust(
    opens: Vec<f64>,
    highs: Vec<f64>,
    lows: Vec<f64>,
    closes: Vec<f64>,
    lookback_period: usize,
    body_multiplier: f64,
    min_gap_val: f64,
) -> PyResult<Vec<Option<(String, f64, f64, usize)>>> {
    let len = opens.len();
    let mut fvg_list: Vec<Option<(String, f64, f64, usize)>> = vec![None, None];

    for i in 2..len {
        let first_high = highs[i - 2];
        let first_low = lows[i - 2];
        let middle_open = opens[i - 1];
        let middle_close = closes[i - 1];
        let third_low = lows[i];
        let third_high = highs[i];

        let middle_body = (middle_close - middle_open).abs();

        // Calculate average body size from previous candles
        let start_idx = if i > lookback_period + 1 {
            i - 1 - lookback_period
        } else {
            0
        };

        let mut sum_bodies = 0.0;
        let mut count = 0;
        for j in start_idx..i - 1 {
            let body = (closes[j] - opens[j]).abs();
            sum_bodies += body;
            count += 1;
        }

        let avg_body_size = if count > 0 {
            sum_bodies / count as f64
        } else {
            0.001
        };

        let min_gap = middle_body * min_gap_val;

        // Check Bullish FVG
        if third_low > first_high {
            let gap_size = third_low - first_high;
            if middle_body > avg_body_size * body_multiplier && gap_size > min_gap {
                fvg_list.push(Some(("bullish".to_string(), first_high, third_low, i)));
                continue;
            }
        }

        // Check Bearish FVG
        if third_high < first_low {
            let gap_size = first_low - third_high;
            if middle_body > avg_body_size * body_multiplier && gap_size > min_gap {
                fvg_list.push(Some(("bearish".to_string(), first_low, third_high, i)));
                continue;
            }
        }

        fvg_list.push(None);
    }

    Ok(fvg_list)
}

/// Detects key support and resistance levels on 1H timeframe
#[pyfunction]
pub fn detect_key_levels_rust(
    highs: Vec<f64>,
    lows: Vec<f64>,
    backcandles: usize,
    test_candles: usize,
) -> PyResult<(Vec<f64>, Vec<f64>)> {
    let len = highs.len();
    let current_candle = len.saturating_sub(1);
    let mut support_levels: Vec<f64> = Vec::new();
    let mut resistance_levels: Vec<f64> = Vec::new();

    let start = if current_candle > backcandles + test_candles {
        current_candle - backcandles - test_candles
    } else {
        0
    };

    for i in start..=current_candle.saturating_sub(test_candles) {
        if i < test_candles || i >= len.saturating_sub(test_candles) {
            continue;
        }

        let high = highs[i];
        let low = lows[i];

        let before_high_idx_start = if i >= test_candles { i - test_candles } else { 0 };
        let after_high_idx_end = (i + test_candles + 1).min(len);

        // Find max high in window
        let mut max_high = high;
        for j in before_high_idx_start..=i.saturating_sub(1) {
            if highs[j] > max_high {
                max_high = highs[j];
            }
        }
        for j in (i + 1)..after_high_idx_end {
            if highs[j] > max_high {
                max_high = highs[j];
            }
        }

        // Find min low in window
        let mut min_low = low;
        for j in before_high_idx_start..=i.saturating_sub(1) {
            if lows[j] < min_low {
                min_low = lows[j];
            }
        }
        for j in (i + 1)..after_high_idx_end {
            if lows[j] < min_low {
                min_low = lows[j];
            }
        }

        // Check if current high is the highest
        if (high - max_high).abs() < 1e-10 {
            resistance_levels.push(high);
        }

        // Check if current low is the lowest
        if (low - min_low).abs() < 1e-10 {
            support_levels.push(low);
        }
    }

    // Remove duplicates and sort
    resistance_levels.sort_by(|a, b| a.partial_cmp(b).unwrap());
    support_levels.sort_by(|a, b| b.partial_cmp(a).unwrap());

    resistance_levels.dedup_by(|a, b| (*a - *b).abs() < 1e-10);
    support_levels.dedup_by(|a, b| (*a - *b).abs() < 1e-10);

    Ok((support_levels, resistance_levels))
}

#[pymodule]
fn trading_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(detect_fvg_rust, m)?)?;
    m.add_function(wrap_pyfunction!(detect_key_levels_rust, m)?)?;
    Ok(())
}
