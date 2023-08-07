use byteorder::{LittleEndian, ReadBytesExt};
use libc::{c_void, mmap, rand, MAP_PRIVATE, PROT_READ, RAND_MAX};
use rayon::iter::ParallelIterator;
use rayon::prelude::{IndexedParallelIterator, IntoParallelIterator, IntoParallelRefMutIterator};
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom, Write};
use std::os::unix::io::IntoRawFd;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Default)]
struct Config {
    dim: i32,
    hidden_dim: i32,
    n_layers: i32,
    n_heads: i32,
    n_kv_heads: i32,
    vocab_size: i32,
    seq_len: i32,
}

#[derive(Debug, Default)]
struct TransformerWeights<'a> {
    token_embedding_table: &'a [f32],
    rms_att_weight: &'a [f32],
    rms_ffn_weight: &'a [f32],
    wq: &'a [f32],
    wk: &'a [f32],
    wv: &'a [f32],
    wo: &'a [f32],
    w1: &'a [f32],
    w2: &'a [f32],
    w3: &'a [f32],
    rms_final_weight: &'a [f32],
    freq_cis_real: &'a [f32],
    freq_cis_imag: &'a [f32],
    wcls: &'a [f32],
}

#[derive(Debug, Default)]
struct RunState {
    x: Vec<f32>,
    xb: Vec<f32>,
    xb2: Vec<f32>,
    hb: Vec<f32>,
    hb2: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    att: Vec<f32>,
    logits: Vec<f32>,
    key_cache: Vec<f32>,
    value_cache: Vec<f32>,
}

// TODO: Need to implement sync trait for a wrapper of the RunState struct.
// struct RunStatePtr(*const RunState);
// unsafe impl Sync for RunStatePtr {}

fn malloc_run_state(s: &mut RunState, p: &Config) {
    s.x = vec![0.0; p.dim as usize];
    s.xb = vec![0.0; p.dim as usize];
    s.xb2 = vec![0.0; p.dim as usize];
    s.hb = vec![0.0; p.hidden_dim as usize];
    s.hb2 = vec![0.0; p.hidden_dim as usize];
    s.q = vec![0.0; p.dim as usize];
    s.k = vec![0.0; p.dim as usize];
    s.v = vec![0.0; p.dim as usize];
    s.att = vec![0.0; (p.n_heads * p.seq_len) as usize];
    s.logits = vec![0.0; p.vocab_size as usize];
    s.key_cache = vec![0.0; (p.n_layers * p.seq_len * p.dim) as usize];
    s.value_cache = vec![0.0; (p.n_layers * p.seq_len * p.dim) as usize];
}

fn time_in_ms() -> u64 {
    let time = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
    return time.as_secs() * 1000 + time.subsec_nanos() as u64 / 1_000_000;
}

fn checkpoint_init_weights<'a>(
    p: &Config,
    f: &'a mut [f32],
    shared_weights: bool,
) -> TransformerWeights<'a> {
    let mut w: TransformerWeights<'a> = TransformerWeights::default();

    let mut ptr = 0;
    w.token_embedding_table = &f[ptr..ptr + p.vocab_size as usize * p.dim as usize];
    ptr += p.vocab_size as usize * p.dim as usize;
    w.rms_att_weight = &f[ptr..ptr + p.n_layers as usize * p.dim as usize];
    ptr += p.n_layers as usize * p.dim as usize;
    w.wq = &f[ptr..ptr + p.n_layers as usize * p.dim as usize * p.dim as usize];
    ptr += p.n_layers as usize * p.dim as usize * p.dim as usize;
    w.wk = &f[ptr..ptr + p.n_layers as usize * p.dim as usize * p.dim as usize];
    ptr += p.n_layers as usize * p.dim as usize * p.dim as usize;
    w.wv = &f[ptr..ptr + p.n_layers as usize * p.dim as usize * p.dim as usize];
    ptr += p.n_layers as usize * p.dim as usize * p.dim as usize;
    w.wo = &f[ptr..ptr + p.n_layers as usize * p.dim as usize * p.dim as usize];
    ptr += p.n_layers as usize * p.dim as usize * p.dim as usize;
    w.rms_ffn_weight = &f[ptr..ptr + p.n_layers as usize * p.dim as usize];
    ptr += p.n_layers as usize * p.dim as usize;
    w.w1 = &f[ptr..ptr + p.n_layers as usize * p.hidden_dim as usize * p.dim as usize];
    ptr += p.n_layers as usize * p.hidden_dim as usize * p.dim as usize;
    w.w2 = &f[ptr..ptr + p.n_layers as usize * p.hidden_dim as usize * p.dim as usize];
    ptr += p.n_layers as usize * p.hidden_dim as usize * p.dim as usize;
    w.w3 = &f[ptr..ptr + p.n_layers as usize * p.hidden_dim as usize * p.dim as usize];
    ptr += p.n_layers as usize * p.hidden_dim as usize * p.dim as usize;
    w.rms_final_weight = &f[ptr..ptr + p.dim as usize];
    ptr += p.dim as usize;
    let head_size = p.dim as usize / p.n_heads as usize;
    w.freq_cis_real = &f[ptr..ptr + p.seq_len as usize * head_size / 2];
    ptr += p.seq_len as usize * head_size / 2;
    w.freq_cis_imag = &f[ptr..ptr + p.seq_len as usize * head_size / 2];
    ptr += p.seq_len as usize * head_size / 2;

    w.wcls = if shared_weights {
        w.token_embedding_table
    } else {
        &f[ptr..]
    };
    return w;
}

fn transformer(
    token: usize,
    pos: usize,
    p: &mut Config,
    run_state: &mut RunState,
    w: &mut TransformerWeights,
) {
    // a few convenience variables
    let dim = p.dim as usize;
    let hidden_dim = p.hidden_dim;
    let head_size = dim / p.n_heads as usize;

    // copy the token embedding into x
    let content_row = &w.token_embedding_table[(token * dim)..(token * dim + dim)];

    run_state.x.copy_from_slice(content_row);

    // TODO: This raw pointer should be used in the
    // (0..p.n_heads as usize).into_par_iter().for_each()...
    // so that it can run in parallel. Right now, the borrow checker isn't
    // letting me access it.
    let run_state_ptr = run_state as *mut RunState;

    let s = unsafe { &mut *run_state_ptr };

    for l in 0..p.n_layers as usize {
        rmsnorm(&mut s.xb, &s.x, &mut &w.rms_att_weight[l * dim..]);

        matmul(&mut s.q, &s.xb, &w.wq[l * dim * dim..], dim, dim);
        matmul(&mut s.k, &s.xb, &w.wk[l * dim * dim..], dim, dim);
        matmul(&mut s.v, &s.xb, &w.wv[l * dim * dim..], dim, dim);

        // println!("{:?}", s.k);
        // break;

        // apply RoPE rotation to the q and k vectors for each head
        for h in 0..p.n_heads as usize {
            // get the q and k vectors for this head
            // // rotate q and k by the freq_cis_real and freq_cis_imag
            for i in (0..head_size).step_by(2) {
                let q0 = s.q[h * head_size + i];

                let q1 = s.q[h * head_size + i + 1];
                let k0 = s.k[h * head_size + i];
                let k1 = s.k[h * head_size + i + 1];
                let fcr = w.freq_cis_real[(pos * head_size / 2) + i / 2];
                let fci = w.freq_cis_imag[(pos * head_size / 2) + i / 2];
                s.q[h * head_size + i] = q0 * fcr - q1 * fci;
                s.q[h * head_size + i + 1] = q0 * fci + q1 * fcr;
                s.k[h * head_size + i] = k0 * fcr - k1 * fci;
                s.k[h * head_size + i + 1] = k0 * fci + k1 * fcr;
            }
        }

        let loff = l * p.seq_len as usize * dim;

        let key_cache_row = &mut s.key_cache[(loff + pos * dim)..];
        let value_cache_row = &mut s.value_cache[(loff + pos * dim)..];
        key_cache_row[..dim].copy_from_slice(&s.k[..dim]);
        value_cache_row[..dim].copy_from_slice(&s.v[..dim]);

        //  #pragma omp parallel -- This should run in parallel, fix the issues with sharing a
        // mutable reference to run_state.

        // let run_state_ptr = RunStatePtr(run_state as *mut RunState);
        // (0..p.n_heads as usize).into_par_iter().for_each(|h| {
        //     // Your code here
        //     let q_index = h * head_size;
        //     let att_index = h * p.seq_len as usize;
        //     let s_ = unsafe { run_state_ptr.0 };

        //     for t in 0..(pos + 1) {
        //         let key_vector_index = loff + t * dim + h * head_size;
        //         let mut score = 0.0f32;
        //         for i in 0..head_size {
        //             score += s.q[i + q_index] * s.key_cache[i + key_vector_index];
        //         }

        //         score /= (head_size as f32).sqrt();
        //         s.att[t + att_index] = score;
        //     }

        //     // ////DEBUG PRINTS
        //     // println!("{:?}", s.att);
        //     // return;

        //     softmax(&mut s.att, att_index, att_index + pos + 1);

        //     for i in 0..head_size {
        //         let mut val = 0.0f32;
        //         for t in 0..(pos + 1) {
        //             val += s.att[att_index + t] * s.value_cache[loff + t * dim + h * head_size + i]
        //         }
        //         s.xb[h * head_size + i] = val
        //     }
        // });

        for h in 0..p.n_heads as usize {
            let q_index = h * head_size;
            let att_index = h * p.seq_len as usize;

            for t in 0..(pos + 1) {
                let key_vector_index = loff + t * dim + h * head_size;
                let mut score = 0.0f32;
                for i in 0..head_size {
                    score += s.q[i + q_index] * s.key_cache[i + key_vector_index];
                }

                score /= (head_size as f32).sqrt();
                s.att[t + att_index] = score;
            }

            softmax(&mut s.att, att_index, att_index + pos + 1);

            for i in 0..head_size {
                let mut val = 0.0f32;
                for t in 0..(pos + 1) {
                    val += s.att[att_index + t] * s.value_cache[loff + t * dim + h * head_size + i]
                }
                s.xb[h * head_size + i] = val
            }
        }

        matmul(&mut s.xb2, &mut s.xb, &w.wo[l * dim * dim..], dim, dim);

        accum(&mut s.x, &s.xb2);

        // ffn rmsnorm, again  This can be an issue. x might not be mutably changed.
        rmsnorm(&mut s.xb, &s.x, &w.rms_ffn_weight[l * dim..]);

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // first calculate self.w1(x) and self.w3(x)
        //matmul(&mut s.v, &s.xb, &w.wv[l * dim * dim..], dim, dim);
        matmul(
            &mut s.hb,
            &s.xb,
            &w.w1[l * dim * hidden_dim as usize..],
            dim,
            hidden_dim as usize,
        );
        matmul(
            &mut s.hb2,
            &s.xb,
            &w.w3[l * dim * hidden_dim as usize..],
            dim,
            hidden_dim as usize,
        );

        // F.silu; silu(x)=x*σ(x),where σ(x) is the logistic sigmoid
        for i in 0..hidden_dim as usize {
            s.hb[i] = s.hb[i] * (1.0f32 / (1.0f32 + ((-s.hb[i]).exp())))
        }

        for i in 0..hidden_dim as usize {
            s.hb[i] = s.hb[i] * s.hb2[i];
        }

        matmul(
            &mut s.xb,
            &s.hb,
            &w.w2[l * dim * hidden_dim as usize..],
            hidden_dim as usize,
            dim,
        );

        accum(&mut s.x, &s.xb);
    }

    self_rmsnorm(&mut s.x, w.rms_final_weight);
    matmul(&mut s.logits, &s.x, &w.wcls, dim, p.vocab_size as usize);
}

fn accum(a: &mut [f32], b: &[f32]) {
    for (a, b) in a.iter_mut().zip(b.iter()) {
        *a += *b;
    }
}

fn self_rmsnorm(o: &mut [f32], weight: &[f32]) {
    let mut ss = 0.0f32;
    for j in 0..o.len() {
        ss += o[j] * o[j];
    }
    ss /= o.len() as f32;
    ss += 1e-5;
    let ss = 1.0 / ss.sqrt();
    for j in 0..o.len() {
        o[j] = weight[j] * (ss * o[j]);
    }
}

fn rmsnorm(o: &mut [f32], x: &[f32], weight: &[f32]) {
    let ss = x.iter().map(|&x| x * x).sum::<f32>() / x.len() as f32 + 1e-5;
    let ss = 1.0 / ss.sqrt();
    for (o, (&x, &weight)) in o.iter_mut().zip(x.iter().zip(weight.iter())) {
        *o = weight * ss * x;
    }
}

fn softmax(x: &mut Vec<f32>, start_index: usize, stop_index: usize) {
    let max_val = x
        .iter()
        .skip(start_index)
        .take(stop_index - start_index)
        .fold(f32::NEG_INFINITY, |a, b| a.max(*b));
    let sum = x
        .iter_mut()
        .skip(start_index)
        .take(stop_index - start_index)
        .map(|x| {
            *x = (*x - max_val).exp();
            *x
        })
        .sum::<f32>();
    for x in x
        .iter_mut()
        .skip(start_index)
        .take(stop_index - start_index)
    {
        *x /= sum;
    }
}

fn matmul(xout: &mut [f32], x: &[f32], w: &[f32], n: usize, _d: usize) {
    // Parallel matrix multiplication.
    xout.par_iter_mut().enumerate().for_each(|(i, xout_i)| {
        let val = (0..n).map(|j| w[i * n + j] * x[j]).sum::<f32>();
        *xout_i = val;
    });
}

fn sample(probabilities: &Vec<f32>) -> usize {
    let r: f32 = unsafe { rand() as f32 / RAND_MAX as f32 };
    let mut cdf: f32 = 0.0;
    for (i, &prob) in probabilities.iter().enumerate() {
        cdf += prob;
        if r < cdf {
            return i;
        }
    }
    probabilities.len() - 1 // in case of rounding errors
}

fn argmax(v: &Vec<f32>) -> usize {
    let (max_i, _) = v
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap();
    max_i
}

fn main() {
    let mut file =
        File::open("./resources/model.bin").expect("Unable to open the checkpoint file!");
    let mut steps = 256;
    let mut config = Config::default();
    file.read_exact(unsafe {
        std::slice::from_raw_parts_mut(
            &mut config as *mut _ as *mut u8,
            std::mem::size_of::<Config>(),
        )
    })
    .expect("Failed to read config!");

    let shared_weights = if config.vocab_size > 0 { true } else { false };
    config.vocab_size = config.vocab_size.abs();

    file.seek(SeekFrom::End(0))
        .expect("Failed to seek to end of file!");
    let file_size = file.metadata().expect("Failed to get metadata!").len();

    let fd = file.into_raw_fd();
    let data = unsafe {
        mmap(
            std::ptr::null_mut(),
            file_size as _,
            PROT_READ,
            MAP_PRIVATE,
            fd,
            0,
        ) as *mut c_void
    };
    if data.is_null() {
        println!("mmap failed!");
    }

    let weights_ptr: *mut c_void = unsafe { data.offset((std::mem::size_of::<Config>()) as isize) };

    let weight_pointer: &mut [f32] = unsafe {
        std::slice::from_raw_parts_mut(
            weights_ptr as *mut _ as *mut f32,
            (file_size as usize - std::mem::size_of::<Config>() as usize),
        )
    };

    let mut weights: TransformerWeights<'_> =
        checkpoint_init_weights(&mut config, weight_pointer, shared_weights);

    // right now we cannot run for more than config.seq_len steps
    if steps <= 0 || steps > config.seq_len {
        steps = config.seq_len;
    }

    let file = File::open("./resources/tokenizer.bin").unwrap();
    let mut reader = BufReader::new(file);
    let mut vocab = Vec::new();

    for _ in 0..config.vocab_size {
        let len = reader.read_u32::<LittleEndian>().unwrap() as usize;
        let mut buffer = vec![0; len];
        reader.read_exact(&mut buffer).unwrap();
        let string = String::from_utf8(buffer).unwrap();
        vocab.push(string);
    }

    // print!("{:?}", vocab);

    let mut run_state = RunState::default();

    malloc_run_state(&mut run_state, &config);

    let start = time_in_ms();
    let mut next = 0;
    let mut token = 1; // 1 = BOS token in Llama-2 sentencepiece
    let mut pos = 0;
    let temperature = 0.9f32; // e.g. 1.0, or 0.0
    println!("<s>\n"); // explicit print the initial BOS token (=1), stylistically symmetric
    while (pos < steps) {
        // forward the transformer to get logits for the next token
        //// DEBUG PRINTS
        //print!("{:?}", run_state.logits[0]);
        transformer(
            token,
            pos as usize,
            &mut config,
            &mut run_state,
            &mut weights,
        );

        // DEBUG PRINTS
        // println!("{:?}", run_state.logits);
        // break;

        if temperature == 0.0f32 {
            // greedy argmax sampling
            next = argmax(&run_state.logits);
        } else {
            for q in 0..config.vocab_size {
                run_state.logits[q as usize] /= temperature;
            }

            softmax(&mut run_state.logits, 0, config.vocab_size as usize);

            next = sample(&run_state.logits);
        }
        print!("{}", vocab[next]);
        std::io::stdout().flush().unwrap();
        // advance forward
        token = next;
        pos += 1;
    }

    let end = time_in_ms();
    println!(
        "\nachieved tok/s: {:?}\n",
        steps as u64 / ((end - start) / 1000)
    );
}
