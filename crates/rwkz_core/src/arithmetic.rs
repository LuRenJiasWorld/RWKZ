/// Arithmetic coding encoder and decoder.
///
/// Uses 32-bit precision with bit-level output.
/// Designed for deterministic, cross-platform behavior.

const PRECISION: u32 = 32;
const WHOLE: u64 = 1u64 << PRECISION;
const HALF: u64 = WHOLE >> 1;
const QUARTER: u64 = HALF >> 1;

/// Arithmetic encoder that writes compressed bits to a byte buffer.
pub struct Encoder {
    low: u64,
    high: u64,
    pending_bits: u32,
    output: Vec<u8>,
    bit_buffer: u8,
    bits_in_buffer: u8,
}

impl Encoder {
    pub fn new() -> Self {
        Self {
            low: 0,
            high: WHOLE - 1,
            pending_bits: 0,
            output: Vec::new(),
            bit_buffer: 0,
            bits_in_buffer: 0,
        }
    }

    /// Encode a symbol given its probability distribution as a CDF.
    ///
    /// `cdf` is a cumulative distribution function array of length (vocab_size + 1).
    /// cdf[i] / cdf[vocab_size] gives the cumulative probability for symbol i.
    /// Symbol `sym` must be in [0, vocab_size).
    pub fn encode_symbol(&mut self, sym: usize, cdf: &[u32]) {
        let total = cdf[cdf.len() - 1] as u64;
        let sym_low = cdf[sym] as u64;
        let sym_high = cdf[sym + 1] as u64;

        let range = self.high - self.low + 1;
        self.high = self.low + (range * sym_high) / total - 1;
        self.low = self.low + (range * sym_low) / total;

        loop {
            if self.high < HALF {
                self.output_bit(false);
                self.low <<= 1;
                self.high = (self.high << 1) | 1;
            } else if self.low >= HALF {
                self.output_bit(true);
                self.low = (self.low - HALF) << 1;
                self.high = ((self.high - HALF) << 1) | 1;
            } else if self.low >= QUARTER && self.high < THREE_QUARTER {
                self.pending_bits += 1;
                self.low = (self.low - QUARTER) << 1;
                self.high = ((self.high - QUARTER) << 1) | 1;
            } else {
                break;
            }
        }
    }

    /// Finalize encoding and return the compressed byte buffer.
    pub fn finish(mut self) -> Vec<u8> {
        self.pending_bits += 1;
        if self.low < QUARTER {
            self.output_bit(false);
        } else {
            self.output_bit(true);
        }
        // Flush remaining bits with padding
        if self.bits_in_buffer > 0 {
            self.bit_buffer <<= 8 - self.bits_in_buffer;
            self.output.push(self.bit_buffer);
        }
        self.output
    }

    fn output_bit(&mut self, bit: bool) {
        self.write_bit(bit);
        for _ in 0..self.pending_bits {
            self.write_bit(!bit);
        }
        self.pending_bits = 0;
    }

    fn write_bit(&mut self, bit: bool) {
        self.bit_buffer <<= 1;
        if bit {
            self.bit_buffer |= 1;
        }
        self.bits_in_buffer += 1;
        if self.bits_in_buffer == 8 {
            self.output.push(self.bit_buffer);
            self.bit_buffer = 0;
            self.bits_in_buffer = 0;
        }
    }
}

const THREE_QUARTER: u64 = HALF + QUARTER;

/// Arithmetic decoder that reads compressed bits from a byte slice.
pub struct Decoder<'a> {
    data: &'a [u8],
    byte_pos: usize,
    bit_pos: u8,
    low: u64,
    high: u64,
    code: u64,
}

impl<'a> Decoder<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        let mut dec = Self {
            data,
            byte_pos: 0,
            bit_pos: 0,
            low: 0,
            high: WHOLE - 1,
            code: 0,
        };
        // Initialize code with first PRECISION bits
        for _ in 0..PRECISION {
            dec.code = (dec.code << 1) | dec.read_bit() as u64;
        }
        dec
    }

    /// Decode one symbol given the CDF.
    ///
    /// Returns the decoded symbol index.
    pub fn decode_symbol(&mut self, cdf: &[u32]) -> usize {
        let total = cdf[cdf.len() - 1] as u64;
        let range = self.high - self.low + 1;
        let scaled_value = ((self.code - self.low + 1) * total - 1) / range;

        // Find symbol via binary search on CDF
        let sym = cdf_partition(cdf, scaled_value as u32);

        let sym_low = cdf[sym] as u64;
        let sym_high = cdf[sym + 1] as u64;
        self.high = self.low + (range * sym_high) / total - 1;
        self.low = self.low + (range * sym_low) / total;

        loop {
            if self.high < HALF {
                self.low <<= 1;
                self.high = (self.high << 1) | 1;
                self.code = (self.code << 1) | self.read_bit() as u64;
            } else if self.low >= HALF {
                self.low = (self.low - HALF) << 1;
                self.high = ((self.high - HALF) << 1) | 1;
                self.code = ((self.code - HALF) << 1) | self.read_bit() as u64;
            } else if self.low >= QUARTER && self.high < THREE_QUARTER {
                self.low = (self.low - QUARTER) << 1;
                self.high = ((self.high - QUARTER) << 1) | 1;
                self.code = ((self.code - QUARTER) << 1) | self.read_bit() as u64;
            } else {
                break;
            }
        }

        sym
    }

    fn read_bit(&mut self) -> bool {
        if self.byte_pos >= self.data.len() {
            return false;
        }
        let bit = (self.data[self.byte_pos] >> (7 - self.bit_pos)) & 1 == 1;
        self.bit_pos += 1;
        if self.bit_pos == 8 {
            self.byte_pos += 1;
            self.bit_pos = 0;
        }
        bit
    }
}

/// Find the symbol whose CDF range contains `value` using binary search.
fn cdf_partition(cdf: &[u32], value: u32) -> usize {
    let mut lo = 0;
    let mut hi = cdf.len() - 2; // last valid symbol index
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if cdf[mid + 1] <= value {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

/// Build a CDF from a probability distribution using f64 precision.
///
/// Returns a CDF of length (probs.len() + 1) with values in [0, 2^31).
/// Each symbol is guaranteed at least 1 unit of CDF range to prevent
/// encoding/decoding ambiguity.
pub fn build_cdf_from_probs(probs: &[f32]) -> Vec<u32> {
    let n = probs.len();
    let max_cdf = 1u64 << 31;

    // Use f64 for accumulation to avoid f32 precision loss
    let total: f64 = probs.iter().map(|&p| p as f64).sum();

    // Build CDF using f64, then quantize to u32
    let mut cdf_f64 = Vec::with_capacity(n + 1);
    cdf_f64.push(0.0f64);
    let mut cumulative: f64 = 0.0;
    for &p in probs {
        cumulative += p as f64 / total;
        cdf_f64.push(cumulative);
    }

    // Quantize to integer CDF, ensuring strict monotonicity
    // Each symbol must have at least 1 unit of range
    let mut cdf = vec![0u32; n + 1];
    let mut prev: u32 = 0;
    for i in 1..=n {
        let val = (cdf_f64[i] * (max_cdf as f64)).floor() as u64;
        let val = val.min(max_cdf - 1);
        let val = val.max((prev as u64) + 1); // strict monotonicity
        cdf[i] = val as u32;
        prev = val as u32;
    }
    cdf[n] = (max_cdf - 1) as u32;
    cdf
}

/// Convert logits to probabilities using softmax (temperature = 1).
pub fn softmax(logits: &[f32]) -> Vec<f32> {
    let max_val = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&x| (x - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|&x| x / sum).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roundtrip_uniform() {
        let vocab_size = 256;
        let cdf: Vec<u32> = (0..=vocab_size)
            .map(|i| ((i as u64 * ((1u64 << 31) - 1)) / vocab_size as u64) as u32)
            .collect();

        let symbols: Vec<usize> = (0..1000).map(|i| (i * 7 + 3) % vocab_size).collect();

        let mut encoder = Encoder::new();
        for &sym in &symbols {
            encoder.encode_symbol(sym, &cdf);
        }
        let compressed = encoder.finish();

        let mut decoder = Decoder::new(&compressed);
        let decoded: Vec<usize> = symbols.iter().map(|_| decoder.decode_symbol(&cdf)).collect();

        assert_eq!(symbols, decoded);
    }

    #[test]
    fn test_roundtrip_skewed_distribution() {
        // Non-uniform distribution: first few symbols very likely
        let probs = vec![0.3, 0.25, 0.2, 0.1, 0.05, 0.04, 0.03, 0.02, 0.01];
        let cdf = build_cdf_from_probs(&probs);

        let symbols = vec![0, 0, 1, 0, 2, 1, 3, 0, 0, 4, 1, 2, 5, 0, 6, 7, 8, 0, 1, 2];

        let mut encoder = Encoder::new();
        for &sym in &symbols {
            encoder.encode_symbol(sym, &cdf);
        }
        let compressed = encoder.finish();

        let mut decoder = Decoder::new(&compressed);
        let decoded: Vec<usize> = symbols.iter().map(|_| decoder.decode_symbol(&cdf)).collect();

        assert_eq!(symbols, decoded);
    }

    #[test]
    fn test_compression_is_smaller() {
        // With a skewed distribution, compressed output should be smaller than raw
        let probs = vec![0.5, 0.25, 0.125, 0.0625, 0.0625];
        let cdf = build_cdf_from_probs(&probs);

        // All symbol 0 (most probable) — should compress very well
        let symbols: Vec<usize> = vec![0; 1000];

        let mut encoder = Encoder::new();
        for &sym in &symbols {
            encoder.encode_symbol(sym, &cdf);
        }
        let compressed = encoder.finish();

        // Each symbol 0 takes ~1 bit with 50% probability, so ~125 bytes for 1000 symbols
        assert!(compressed.len() < 200, "compressed {} bytes", compressed.len());
    }

    #[test]
    fn test_softmax() {
        let logits = vec![1.0, 2.0, 3.0];
        let probs = softmax(&logits);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        assert!(probs[2] > probs[1]);
        assert!(probs[1] > probs[0]);
    }

    #[test]
    fn test_cdf_partition() {
        let cdf = vec![0, 100, 300, 600, 900, 1000];
        assert_eq!(cdf_partition(&cdf, 50), 0);
        assert_eq!(cdf_partition(&cdf, 150), 1);
        assert_eq!(cdf_partition(&cdf, 450), 2);
        assert_eq!(cdf_partition(&cdf, 750), 3);
        assert_eq!(cdf_partition(&cdf, 950), 4);
    }
}
