"""
prepare_training_data.py — builds the fine-tuning dataset for text-to-SQL.

Source: gretelai/synthetic_text_to_sql  (Apache 2.0, 100k train records)
  Each record has:
    sql_prompt   -> the natural-language question
    sql_context  -> CREATE TABLE statements (the schema)
    sql          -> the gold SQL answer

We convert each record into the CHAT / INSTRUCTION format that an *instruct*
model (Qwen2.5-7B-Instruct) was trained to follow. That format is the key:
the model already knows how to follow system/user/assistant turns, so we
present our task in that same shape and it learns the question->SQL mapping.

  ┌─────────────────────────────────────────────────────────────┐
  │ CONCEPT: why the chat format matters                        │
  │                                                             │
  │ A base model just continues text. An *instruct* model was   │
  │ post-trained on (system, user, assistant) conversations.    │
  │ To fine-tune it effectively you must speak that same        │
  │ language — otherwise you fight the model's training instead  │
  │ of building on it. Each of our examples becomes:            │
  │                                                             │
  │   system:    "You are an expert text-to-SQL..."             │
  │   user:      "<schema>\n\nQuestion: <nl question>"          │
  │   assistant: "<gold sql>"                                    │
  │                                                             │
  │ During training, the model is optimized to produce the      │
  │ assistant turn given the system+user turns. The loss is     │
  │ computed ONLY on the assistant tokens (the SQL) — we don't   │
  │ want it learning to generate questions or schemas.          │
  └─────────────────────────────────────────────────────────────┘

Output: a JSONL file of chat-formatted examples, ready for TRL's SFTTrainer.

Run on your machine (needs internet to HF the first time):
    pip install datasets
    python prepare_training_data.py --n 5000 --out train.jsonl
"""
import argparse
import json


SYSTEM_PROMPT = (
    "You are an expert at converting natural language questions into SQL queries. "
    "Given a database schema and a question, respond with a single correct SQL query "
    "and nothing else."
)


def to_chat_example(sql_prompt: str, sql_context: str, sql: str) -> dict:
    """Convert one raw record into a chat-format training example."""
    user_content = f"{sql_context.strip()}\n\nQuestion: {sql_prompt.strip()}"
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": sql.strip()},
        ]
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000,
                    help="how many training examples to prepare (subset for a first run)")
    ap.add_argument("--out", default="train.jsonl")
    ap.add_argument("--split", default="train", help="dataset split to pull")
    args = ap.parse_args()

    from datasets import load_dataset  # imported here so the file is inspectable without the dep

    print(f"Loading gretelai/synthetic_text_to_sql (split={args.split}) ...")
    ds = load_dataset("gretelai/synthetic_text_to_sql", split=args.split)
    print(f"  full split size: {len(ds):,}")

    n = min(args.n, len(ds))
    ds = ds.select(range(n))
    print(f"  using first {n:,} records")

    written = 0
    with open(args.out, "w") as f:
        for rec in ds:
            ex = to_chat_example(rec["sql_prompt"], rec["sql_context"], rec["sql"])
            f.write(json.dumps(ex) + "\n")
            written += 1

    print(f"Wrote {written:,} chat-format examples -> {args.out}")
    print("\nSample example:")
    print(json.dumps(to_chat_example(
        ds[0]["sql_prompt"], ds[0]["sql_context"], ds[0]["sql"]
    ), indent=2)[:800])


if __name__ == "__main__":
    main()
