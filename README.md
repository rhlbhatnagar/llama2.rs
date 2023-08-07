## llama2.c

<p align="center">
  <img src="assets/llama-rust.png" width="300" height="300" alt="Llama with a crab">
</p>

## Rust meets llama.

A mimimal Rust implementation of [karpathy's](https://github.com/karpathy) [llama.c](https://github.com/karpathy/llama2.c).


Currently the code uses the 15M parameter model provided by Karpathy (included in the resources folder). But you should be able to replace that with any llama model. You can read the (section here)[https://github.com/karpathy/llama2.c#metas-llama-2-models] to download larger models.


## Performance:
Right now I'm getting similiar performance on my M1 Macbook for llama.c and llama.rs (~120 tok/s). Though I think we can unlock a lot of performance benifits by parallelising some parts of the code. Left some comments in main.rs on where we can make these gains. I'm no expert on Rust, so PRs are always welcome.


## TODO: 

- Support for quantized versions, 16 bit / 4 bit.
- More parallelization.
- Other improvements like taking in the temp / starting completion string / model path as command line args.


```
# Development
> cargo run

# Prod
> cargo build --release && ./target/release/llama2rs

```
