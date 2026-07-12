"""
finetune.py — LoRA/QLoRA fine-tuning of Qwen2.5-7B-Instruct for text-to-SQL.

Runs on the A10 GPU (24GB). Produces a small LoRA adapter (~tens of MB) that
is served ON TOP OF the frozen base model — you never modify the 7B weights.

────────────────────────────────────────────────────────────────────────────
 CONCEPT MAP (read this before running)
────────────────────────────────────────────────────────────────────────────

1. WHY NOT FULL FINE-TUNING?
   Updating all 7B params needs the model + gradients + optimizer states in
   GPU memory at once — roughly 4x model size (~60GB+). The A10 has 24GB.
   Full fine-tuning simply won't fit.

2. LoRA (Low-Rank Adaptation)
   Freeze the entire base model. For selected weight matrices W, inject a
   small pair of matrices A (d×r) and B (r×d) with rank r << d. The effective
   weight becomes  W + (B·A)·(alpha/r).  You train ONLY A and B — often <1%
   of parameters. The intuition: the *change* needed to specialize a model is
   low-rank, so a compact adapter captures it. Output = tiny adapter file.

   Key knobs:
     r (rank)          — capacity of the adapter. 8–64 typical. Higher = more
                         expressive but more params. We use 16.
     lora_alpha        — scaling; effective LR on the adapter. Common: 2*r.
     target_modules    — which matrices get adapters (attention q/k/v/o +
                         MLP projections is a strong default).
     lora_dropout      — regularization on the adapter.

3. QLoRA (Quantized LoRA)
   Load the FROZEN base model in 4-bit (nf4) to shrink its memory ~4x, while
   the LoRA adapters train in higher precision. This is what gives comfortable
   headroom on a 24GB A10 for a 7B model. Tiny quality cost, big memory win.

4. WHAT GETS OPTIMIZED
   SFTTrainer computes loss only on the ASSISTANT tokens (the SQL). The model
   learns "given this schema+question, produce this SQL" — not to generate
   questions or schemas.

────────────────────────────────────────────────────────────────────────────
 Run (on the GPU node / a GPU box):
   pip install torch transformers peft trl datasets accelerate bitsandbytes
   python finetune.py --data train.jsonl --out ./qwen-sql-lora --epochs 1
────────────────────────────────────────────────────────────────────────────
"""
import argparse
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="train.jsonl", help="chat-format JSONL from prepare_training_data.py")
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--out", default="./qwen-sql-lora")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--max-seq-len", type=int, default=1024)
    ap.add_argument("--qlora", action="store_true", help="use 4-bit QLoRA (recommended on A10)")
    args = ap.parse_args()

    import torch
    from datasets import load_dataset
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer, SFTConfig

    # ── tokenizer ────────────────────────────────────────────────────────────
    tok = AutoTokenizer.from_pretrained(args.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token  # Qwen has no pad token by default

    # ── base model (optionally 4-bit for QLoRA) ──────────────────────────────
    quant_config = None
    if args.qlora:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",             # normalized float 4 — best for LLMs
            bnb_4bit_compute_dtype=torch.bfloat16,  # compute in bf16, store in 4-bit
            bnb_4bit_use_double_quant=True,         # extra memory saving
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    if args.qlora:
        model = prepare_model_for_kbit_training(model)

    # ── LoRA config ──────────────────────────────────────────────────────────
    lora = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank * 2,          # common heuristic: alpha = 2*r
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[                   # attention + MLP projections
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()     # shows the <1% trainable figure

    # ── data ─────────────────────────────────────────────────────────────────
    ds = load_dataset("json", data_files=args.data, split="train")
    print(f"Training examples: {len(ds):,}")

    # ── training ─────────────────────────────────────────────────────────────
    cfg = SFTConfig(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,   # effective batch = bs * accum
        learning_rate=args.lr,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        max_length=args.max_seq_len,
        packing=False,                     # keep examples separate (cleaner for eval parity)
        report_to="none",                  # we'll wire MLflow in the next phase
    )

    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=ds,
        processing_class=tok,
    )

    print("\nStarting training…")
    trainer.train()

    trainer.save_model(args.out)           # saves the LoRA adapter only
    tok.save_pretrained(args.out)
    print(f"\nSaved LoRA adapter -> {args.out}")
    print("Serve it with vLLM via --enable-lora --lora-modules sql-lora=" + args.out)


if __name__ == "__main__":
    main()
