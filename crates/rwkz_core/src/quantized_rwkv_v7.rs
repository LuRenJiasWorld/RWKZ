//! Quantized RWKV v7 model implementation.
//!
//! Same architecture as rwkv_v7 but loads quantized weights (Q8_0/Q4_0 etc.)
//! from GGUF files for reduced download size. Weights are dequantized to F32
//! at load time — for small models (~100MB) this gives much faster CPU
//! inference than keeping them quantized.
//!
//! Only supports base V7 variant (no v7a/v7b DeepEmbed/DEA).
//!
//! GGUF tensor naming convention:
//!   blk.{i}.time_mix_*     — attention (TimeMix) parameters
//!   blk.{i}.channel_mix_*  — FFN (ChannelMix) parameters
//!   blk.{i}.attn_norm.*    — layer norm 1
//!   blk.{i}.attn_norm_2.*  — layer norm 2
//!   token_embd.weight      — embedding
//!   token_embd_norm.*      — pre-norm (layer 0 only)
//!   output_norm.*          — final layer norm
//!   output.weight          — output head

use candle_transformers::quantized_var_builder::VarBuilder;
use candle_core::{DType, Result, Tensor};
use candle_nn::Module;

pub use candle_transformers::models::rwkv_v7::{Config, ModelVersion, State, StatePerLayer};
pub use candle_transformers::models::rwkv_v5::Tokenizer;

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Linear layer without bias, loaded from GGUF.
/// For small RWKV models, dequantizing to F32 at load time gives much faster
/// inference on CPU (standard BLAS matmul vs slow Q8_0 kernel) while keeping
/// the download size small.
#[derive(Debug, Clone)]
struct QLinear {
    weight: Tensor,
}

impl QLinear {
    fn new(vb: &VarBuilder, name: &str) -> Result<Self> {
        let ws = vb.get_no_shape(&format!("{name}.weight"))?;
        let weight = ws.dequantize(vb.device())?;
        Ok(Self { weight })
    }
}

impl Module for QLinear {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let w = match *xs.dims() {
            [b1, b2, _, _] => self.weight.broadcast_left((b1, b2))?.t()?,
            [bsize, _, _] => self.weight.broadcast_left(bsize)?.t()?,
            _ => self.weight.t()?,
        };
        xs.matmul(&w)
    }
}

fn layer_norm(xs: &Tensor, weight: &Tensor, bias: &Tensor, eps: f64) -> Result<Tensor> {
    let xs_dtype = xs.dtype();
    let needs_conversion = xs_dtype != DType::F32;
    let xs_f32 = if needs_conversion {
        xs.to_dtype(DType::F32)?
    } else {
        xs.clone()
    };
    let dim = xs_f32.dim(candle_core::D::Minus1)?;
    let mean = (xs_f32.sum_keepdim(candle_core::D::Minus1)? / dim as f64)?;
    let centered = xs_f32.broadcast_sub(&mean)?;
    let var = (centered.sqr()?.sum_keepdim(candle_core::D::Minus1)? / dim as f64)?;
    let xs = centered.broadcast_div(&(var + eps)?.sqrt()?)?;
    let xs = if needs_conversion {
        xs.to_dtype(xs_dtype)?
    } else {
        xs
    };
    xs.broadcast_mul(weight)?.broadcast_add(bias)
}

/// Infer LoRA dimensions from weight shapes in GGUF.
/// GGUF weight shapes are [out_dim, in_dim], so dim[1] = in_dim.
fn infer_lora_dims(vb: &VarBuilder) -> Result<(usize, usize, usize, usize)> {
    let vb0 = vb.pp("blk").pp(0);
    // GGUF shapes: [out_dim, in_dim], so dims()[1] is the input dimension
    // w1 maps hidden_size -> d_decay, so shape is [d_decay, hidden_size]
    let d_decay = vb0.get_no_shape("time_mix_w1.weight")?.shape().dims()[0];
    let d_aaa = vb0.get_no_shape("time_mix_a1.weight")?.shape().dims()[0];
    let d_mv = vb0.get_no_shape("time_mix_v1.weight")?.shape().dims()[0];
    let d_gate = vb0.get_no_shape("time_mix_g1.weight")?.shape().dims()[0];
    Ok((d_decay, d_aaa, d_mv, d_gate))
}

// ─── TimeMix (Attention) ─────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct TimeMix {
    x_r: Tensor,
    x_w: Tensor,
    x_k: Tensor,
    x_v: Tensor,
    x_a: Tensor,
    x_g: Tensor,
    w0: Tensor,
    w1: QLinear,
    w2: QLinear,
    a0: Tensor,
    a1: QLinear,
    a2: QLinear,
    v0: Option<Tensor>,
    v1: Option<QLinear>,
    v2: Option<QLinear>,
    g1: QLinear,
    g2: QLinear,
    k_k: Tensor,
    k_a: Tensor,
    r_k: Tensor,
    receptance: QLinear,
    key: QLinear,
    value: QLinear,
    output: QLinear,
    ln_x_weight: Tensor,
    ln_x_bias: Tensor,
    layer_id: usize,
    n_heads: usize,
    head_size: usize,
}

impl TimeMix {
    fn new(
        layer_id: usize,
        cfg: &Config,
        lora: (usize, usize, usize, usize),
        vb: VarBuilder,
    ) -> Result<Self> {
        let (_d_decay, _d_aaa, _d_mv, _d_gate) = lora;
        let n_heads = cfg.hidden_size / cfg.head_size;
        let head_size = cfg.head_size;
        let dev = vb.device();

        let dq = |vb: &VarBuilder, name: &str| -> Result<Tensor> {
            vb.get_no_shape(name)?.dequantize(dev)
        };

        // Fused lerp: time_mix_lerp_fused.weight has shape [hidden_size, 1, 1, 6]
        // Split into 6 vectors: x_r, x_w, x_k, x_v, x_a, x_g
        let fused = vb.get_no_shape("time_mix_lerp_fused.weight")?.dequantize(dev)?;
        // fused shape: [768, 1, 1, 6] -> reshape to [768, 6] -> transpose -> [6, 768]
        let c = cfg.hidden_size;
        let fused = fused.reshape((c, 6))?.t()?; // [6, 768]
        let x_r = fused.get(0)?;
        let x_w = fused.get(1)?;
        let x_k = fused.get(2)?;
        let x_v = fused.get(3)?;
        let x_a = fused.get(4)?;
        let x_g = fused.get(5)?;

        let w0 = dq(&vb, "time_mix_w0.weight")?;
        let w1 = QLinear::new(&vb, "time_mix_w1")?;
        let w2 = QLinear::new(&vb, "time_mix_w2")?;

        let a0 = dq(&vb, "time_mix_a0.weight")?;
        let a1 = QLinear::new(&vb, "time_mix_a1")?;
        let a2 = QLinear::new(&vb, "time_mix_a2")?;

        let (v0, v1, v2) = if layer_id > 0 {
            (
                Some(dq(&vb, "time_mix_v0.weight")?),
                Some(QLinear::new(&vb, "time_mix_v1")?),
                Some(QLinear::new(&vb, "time_mix_v2")?),
            )
        } else {
            let _ = vb.get_no_shape("time_mix_v0.weight");
            let _ = vb.get_no_shape("time_mix_v1.weight");
            let _ = vb.get_no_shape("time_mix_v2.weight");
            (None, None, None)
        };

        let g1 = QLinear::new(&vb, "time_mix_g1")?;
        let g2 = QLinear::new(&vb, "time_mix_g2")?;

        let k_k = dq(&vb, "time_mix_k_k.weight")?;
        let k_a = dq(&vb, "time_mix_k_a.weight")?;
        let r_k = dq(&vb, "time_mix_r_k.weight")?;

        let receptance = QLinear::new(&vb, "time_mix_receptance")?;
        let key = QLinear::new(&vb, "time_mix_key")?;
        let value = QLinear::new(&vb, "time_mix_value")?;
        let output = QLinear::new(&vb, "time_mix_output")?;

        let ln_x_weight = dq(&vb, "time_mix_ln.weight")?;
        let ln_x_bias = dq(&vb, "time_mix_ln.bias")?;

        Ok(Self {
            x_r, x_w, x_k, x_v, x_a, x_g,
            w0, w1, w2,
            a0, a1, a2,
            v0, v1, v2,
            g1, g2,
            k_k, k_a, r_k,
            receptance, key, value, output,
            ln_x_weight, ln_x_bias,
            layer_id, n_heads, head_size,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        state: &mut StatePerLayer,
        v_first: Option<Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        let h = self.n_heads;
        let n = self.head_size;

        // 1. Token shift
        let xx = (&state.att_x_prev - x)?;
        let xr = (x + xx.broadcast_mul(&self.x_r)?)?;
        let xw = (x + xx.broadcast_mul(&self.x_w)?)?;
        let xk = (x + xx.broadcast_mul(&self.x_k)?)?;
        let xv = (x + xx.broadcast_mul(&self.x_v)?)?;
        let xa = (x + xx.broadcast_mul(&self.x_a)?)?;
        let xg = (x + xx.broadcast_mul(&self.x_g)?)?;
        state.att_x_prev = x.clone();

        // 2. Linear projections (quantized matmul)
        let r = self.receptance.forward(&xr.unsqueeze(0)?)?.squeeze(0)?;
        let k = self.key.forward(&xk.unsqueeze(0)?)?.squeeze(0)?;
        let v = self.value.forward(&xv.unsqueeze(0)?)?.squeeze(0)?;

        // 3. Decay: w = exp(-0.606531 * sigmoid(w0 + tanh(xw @ w1) @ w2))
        let w = self.w1.forward(&xw.unsqueeze(0)?)?.squeeze(0)?.tanh()?;
        let w = self.w2.forward(&w.unsqueeze(0)?)?.squeeze(0)?;
        let w = (&self.w0 + &w)?;
        let w = w.to_dtype(DType::F32)?;
        let w = (w.neg()?.exp()? + 1.0)?.recip()?;
        let w = (w * (-0.606531))?.exp()?;

        // 4. Value residual
        let (v, v_first) = if self.layer_id == 0 {
            let v_first = v.clone();
            (v, v_first)
        } else {
            let v_first = v_first.unwrap();
            if let (Some(v0), Some(v1), Some(v2)) = (&self.v0, &self.v1, &self.v2) {
                let vw = v1.forward(&xv.unsqueeze(0)?)?.squeeze(0)?;
                let vw2 = v2.forward(&vw.unsqueeze(0)?)?.squeeze(0)?;
                let gate = candle_nn::ops::sigmoid(&(v0 + &vw2)?)?;
                let diff = (&v_first - &v)?.broadcast_mul(&gate)?;
                let v = (&v + &diff)?;
                (v, v_first)
            } else {
                (v, v_first)
            }
        };

        // 5. ICL rate
        let aw = self.a1.forward(&xa.unsqueeze(0)?)?.squeeze(0)?;
        let aw2 = self.a2.forward(&aw.unsqueeze(0)?)?.squeeze(0)?;
        let a = candle_nn::ops::sigmoid(&(&self.a0 + &aw2)?)?;

        // 6. Gate
        let gw = candle_nn::ops::sigmoid(&self.g1.forward(&xg.unsqueeze(0)?)?.squeeze(0)?)?;
        let g = self.g2.forward(&gw.unsqueeze(0)?)?.squeeze(0)?;

        // 7. Key processing
        let kk = (&k * &self.k_k)?;
        let kk = kk.reshape((h, n))?;
        let kk_norm = (kk.sqr()?.sum_keepdim(1)?.sqrt()? + 1e-12)?;
        let kk = kk.broadcast_div(&kk_norm)?;
        let kk = kk.reshape(h * n)?;
        let k_scale = (1.0 + (&a - 1.0)?.broadcast_mul(&self.k_a)?)?;
        let k = (&k * k_scale)?;

        // 8. State update (delta-rule core)
        let v_hn = v.reshape((h, n, 1))?;
        let k_hn = k.reshape((h, 1, n))?;
        let vk = v_hn.matmul(&k_hn)?;

        let kk_h = kk.reshape((h, n))?;
        let a_h = a.reshape((h, n))?;
        let neg_kk = kk_h.neg()?.reshape((h, n, 1))?;
        let kk_a = (&kk_h * &a_h)?.reshape((h, 1, n))?;
        let ab = neg_kk.matmul(&kk_a)?;

        let w_h = w.reshape((h, 1, n))?;
        let att_kv = &state.att_kv;
        let new_state = att_kv.broadcast_mul(&w_h)?
            + att_kv.to_dtype(DType::F32)?.matmul(&ab.to_dtype(DType::F32)?)?
            + vk.to_dtype(DType::F32)?;
        state.att_kv = new_state?;

        let r_hn = r.reshape((h, n, 1))?;
        let out = state.att_kv.to_dtype(r.dtype())?.matmul(&r_hn)?;

        // 9. GroupNorm
        let out = {
            let reshaped = out.reshape((h, n))?;
            let mean = reshaped.mean_keepdim(1)?;
            let centered = reshaped.broadcast_sub(&mean)?;
            let var = centered.sqr()?.mean_keepdim(1)?;
            let normed = centered.broadcast_div(&(var + 64e-5)?.sqrt()?)?;
            normed.reshape(h * n)?
        };
        let out = out.broadcast_mul(&self.ln_x_weight)?.broadcast_add(&self.ln_x_bias)?;

        // 10. Bonus term
        let bonus = (&r * &k * &self.r_k)?
            .reshape((h, n))?
            .sum_keepdim(1)?
            .broadcast_mul(&v.reshape((h, n))?)?
            .reshape(h * n)?;
        let out = (&out + &bonus)?;

        // 11. Output (quantized matmul)
        let gated = (out * g)?;
        let out = self.output.forward(&gated.unsqueeze(0)?)?.squeeze(0)?;

        Ok((out, v_first))
    }
}

// ─── ChannelMix (FFN) ────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct ChannelMix {
    x_k: Tensor,
    key: QLinear,
    value: QLinear,
}

impl ChannelMix {
    fn new(_layer_id: usize, _cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let dev = vb.device();

        let x_k = vb.get_no_shape("channel_mix_lerp_k.weight")?.dequantize(dev)?;
        let key = QLinear::new(&vb, "channel_mix_key")?;
        let value = QLinear::new(&vb, "channel_mix_value")?;

        Ok(Self { x_k, key, value })
    }

    fn forward(&self, x: &Tensor, state: &mut StatePerLayer) -> Result<Tensor> {
        let xx = (&state.ffn_x_prev - x)?;
        let k = (x + xx.broadcast_mul(&self.x_k)?)?;
        state.ffn_x_prev = x.clone();

        let k = self.key.forward(&k.unsqueeze(0)?)?.squeeze(0)?.relu()?.sqr()?;
        Ok(self.value.forward(&k.unsqueeze(0)?)?.squeeze(0)?)
    }
}

// ─── Block ───────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct Block {
    ln1_weight: Tensor,
    ln1_bias: Tensor,
    ln2_weight: Tensor,
    ln2_bias: Tensor,
    att: TimeMix,
    ffn: ChannelMix,
    layer_id: usize,
}

impl Block {
    fn new(
        layer_id: usize,
        cfg: &Config,
        lora: (usize, usize, usize, usize),
        vb: VarBuilder,
    ) -> Result<Self> {
        let dev = vb.device();
        let dq = |vb: &VarBuilder, name: &str| -> Result<Tensor> {
            vb.get_no_shape(name)?.dequantize(dev)
        };

        let ln1_weight = dq(&vb, "attn_norm.weight")?;
        let ln1_bias = dq(&vb, "attn_norm.bias")?;
        let ln2_weight = dq(&vb, "attn_norm_2.weight")?;
        let ln2_bias = dq(&vb, "attn_norm_2.bias")?;

        let att = TimeMix::new(layer_id, cfg, lora, vb.clone())?;
        let ffn = ChannelMix::new(layer_id, cfg, vb)?;

        Ok(Self {
            ln1_weight, ln1_bias,
            ln2_weight, ln2_bias,
            att, ffn,
            layer_id,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        state: &mut State,
        v_first: Option<Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        // ln0 (pre-norm) is applied in Model::forward before entering blocks
        let x_ln1 = layer_norm(x, &self.ln1_weight, &self.ln1_bias, 1e-5)?;
        let (att_out, v_first) = self.att.forward(&x_ln1, &mut state.per_layer[self.layer_id], v_first)?;

        let x = (x + att_out)?;

        let x_ln2 = layer_norm(&x, &self.ln2_weight, &self.ln2_bias, 1e-5)?;
        let ffn_out = self.ffn.forward(&x_ln2, &mut state.per_layer[self.layer_id])?;
        let x = (x + ffn_out)?;

        Ok((x, v_first))
    }
}

// ─── Model ───────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct Model {
    embeddings: candle_nn::Embedding,
    blocks: Vec<Block>,
    ln0_weight: Option<Tensor>,
    ln0_bias: Option<Tensor>,
    ln_out_weight: Tensor,
    ln_out_bias: Tensor,
    head: QLinear,
}

impl Model {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let dev = vb.device();
        let lora = infer_lora_dims(&vb)?;

        // GGUF: candle reverses dims. Raw GGUF stores [vocab, hidden], after reverse [vocab, hidden] (same).
        let emb_weight = vb.get_no_shape("token_embd.weight")?.dequantize(dev)?;
        let embeddings = candle_nn::Embedding::new(emb_weight, cfg.hidden_size);

        // GGUF: token_embd_norm.{weight,bias}
        let ln0_weight = vb.get_no_shape("token_embd_norm.weight")?.dequantize(dev).ok();
        let ln0_bias = vb.get_no_shape("token_embd_norm.bias")?.dequantize(dev).ok();

        let mut blocks = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_blk = vb.pp("blk");
        for layer_id in 0..cfg.num_hidden_layers {
            blocks.push(Block::new(layer_id, cfg, lora, vb_blk.pp(layer_id))?);
        }

        // GGUF: output_norm.{weight,bias}
        let ln_out_weight = vb.get_no_shape("output_norm.weight")?.dequantize(dev)?;
        let ln_out_bias = vb.get_no_shape("output_norm.bias")?.dequantize(dev)?;

        // GGUF: output.weight [vocab, hidden]
        let head = QLinear::new(&vb, "output")?;

        Ok(Self {
            embeddings,
            blocks,
            ln0_weight,
            ln0_bias,
            ln_out_weight,
            ln_out_bias,
            head,
        })
    }

    pub fn forward(&self, xs: &Tensor, state: &mut State, _token_ids: &[u32]) -> Result<Tensor> {
        let xs_emb = xs.apply(&self.embeddings)?;
        let mut xs = xs_emb.squeeze(0)?.squeeze(0)?;

        if let (Some(w), Some(b)) = (&self.ln0_weight, &self.ln0_bias) {
            xs = layer_norm(&xs, w, b, 1e-5)?;
        }

        let mut v_first: Option<Tensor> = None;
        for block in &self.blocks {
            let (new_xs, new_v_first) = block.forward(&xs, state, v_first)?;
            xs = new_xs;
            v_first = Some(new_v_first);
        }

        let xs = layer_norm(&xs, &self.ln_out_weight, &self.ln_out_bias, 1e-5)?;
        let xs = self.head.forward(&xs.unsqueeze(0)?)?.squeeze(0)?;
        state.pos += 1;
        Ok(xs)
    }
}
