# Cog Playground

I'm trying to get familiar with Cog, so this repo serves as a backup of my learning journey.

## Todo

- [x] Build a simple GPT inference pipeline.
- [x] Build a streaming GPT pipeline.
- [ ] Implement a Stable Diffusion pipeline.
    - [ ] Add tracing and model warmup.

## Usage

### GPT2

Basic inference:

```bash
cd nlp/basic
cog predict -i prompt="Hello!" -i max_length=100
```

Streaming inference:

```bash
cd nlp/streaming
cog predict -i prompt="Count form one to twenty: One," -i max_length=100
```