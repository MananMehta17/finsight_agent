"""LoRA fine tuning of DistilBERT for financial sentiment. RUN THIS IN GOOGLE COLAB.

Colab steps:
  1. Runtime > Change runtime type > T4 GPU
  2. Upload this file, then in a cell run:
       !pip install -q peft transformers datasets scikit-learn accelerate
       !python train_sentiment_lora.py
  3. Download finsent_model.zip, unzip it into models/finsent_model in the project.

What LoRA does (interview answer):
  Instead of updating all ~66M DistilBERT weights, we freeze them and add small
  low rank matrices (A and B, rank r) next to the attention query and value layers.
  Only those + the classifier head are trained, roughly 1 to 2% of parameters.
  Faster, cheaper, and the base model knowledge is not destroyed.

NOTE: the dataset name below is a public Hugging Face dataset of labelled finance
tweets. Hugging Face datasets can be renamed or changed, so the script prints the
columns and label names first. If it fails, pick another financial sentiment
dataset with a text column and a 3 class label column and edit DATASET/TEXT_COL.
"""
import numpy as np
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import accuracy_score, f1_score
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments, pipeline)

BASE_MODEL = "distilbert-base-uncased"
DATASET = "zeroshot/twitter-financial-news-sentiment"
TEXT_COL, LABEL_COL = "text", "label"
OUT_DIR = "finsent_model"
SEED = 42

# 1. data ---------------------------------------------------------------
ds = load_dataset(DATASET)
print(ds)
if "validation" not in ds:
    ds = ds["train"].train_test_split(test_size=0.2, seed=SEED)
    ds["validation"] = ds.pop("test")

feat = ds["train"].features[LABEL_COL]
if hasattr(feat, "names"):
    label_names = [n.lower() for n in feat.names]
else:
    # VERIFY this order on the dataset card before trusting the labels
    label_names = ["bearish", "bullish", "neutral"]
    print("WARNING: label names not stored in dataset, assumed:", label_names)
# map market words to the words the agent uses
friendly = {"bearish": "negative", "bullish": "positive"}
label_names = [friendly.get(n, n) for n in label_names]
id2label = dict(enumerate(label_names))
label2id = {v: k for k, v in id2label.items()}
print("Labels:", id2label)

tok = AutoTokenizer.from_pretrained(BASE_MODEL)
def tokenize(batch):
    return tok(batch[TEXT_COL], truncation=True, max_length=128)
ds = ds.map(tokenize, batched=True)

# 2. model + LoRA -------------------------------------------------------
model = AutoModelForSequenceClassification.from_pretrained(
    BASE_MODEL, num_labels=len(label_names), id2label=id2label, label2id=label2id)

lora_cfg = LoraConfig(
    task_type=TaskType.SEQ_CLS,        # keeps the classifier head trainable
    r=8,                               # rank of the low rank matrices
    lora_alpha=16,                     # scaling factor
    lora_dropout=0.1,
    target_modules=["q_lin", "v_lin"], # DistilBERT's attention query and value layers
)
model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()

# 3. train --------------------------------------------------------------
def metrics(pred):
    y = pred.label_ids
    p = np.argmax(pred.predictions, axis=-1)
    return {"accuracy": accuracy_score(y, p), "macro_f1": f1_score(y, p, average="macro")}

args = TrainingArguments(
    output_dir="checkpoints",
    num_train_epochs=3,
    learning_rate=2e-4,                # LoRA usually likes a higher LR than full fine tuning
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    logging_steps=50,
    seed=SEED,
    report_to="none",
)
trainer = Trainer(model=model, args=args, train_dataset=ds["train"],
                  eval_dataset=ds["validation"], processing_class=tok,
                  data_collator=DataCollatorWithPadding(tok), compute_metrics=metrics)
trainer.train()
final = trainer.evaluate()
print("Validation:", final)   # write these numbers in your README / CV, only these

# 4. merge LoRA into the base model and save a normal model folder -------
merged = model.merge_and_unload()
merged.save_pretrained(OUT_DIR)
tok.save_pretrained(OUT_DIR)

# 5. sanity check: reload from disk exactly like the app will ------------
clf = pipeline("text-classification", model=OUT_DIR, tokenizer=OUT_DIR)
for s in ["Company beats earnings estimates and raises guidance",
          "Shares plunge after regulator opens fraud probe",
          "Board meeting scheduled for next Tuesday"]:
    print(s, "->", clf(s)[0])

import shutil
shutil.make_archive(OUT_DIR, "zip", OUT_DIR)
print(f"Done. Download {OUT_DIR}.zip and unzip into models/finsent_model")
