import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import Dataset
from transformers import (
    BertTokenizerFast,
    BertConfig,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments
)

# Dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Tokenizer personalizzato
special_tokens = {
    "unk_token": "[UNK]",
    "pad_token": "[PAD]",
    "cls_token": "[CLS]",
    "sep_token": "[SEP]",
    "mask_token": "[MASK]"
}

tokenizer = BertTokenizerFast(vocab_file="minimal_vocab.txt", **special_tokens)


# Configurazione Mini-BERT
config = BertConfig(
    vocab_size=len(tokenizer),
    hidden_size=256,                 # da 128 → 256
    num_hidden_layers=4,            # da 2 → 4
    num_attention_heads=4,          # da 2 → 4 (deve dividere hidden_size)
    intermediate_size=512,          # da 256 → 512
    max_position_embeddings=128,
    num_labels=2,
    position_embedding_type="absolute"
)


# Modello da zero
model = BertForSequenceClassification(config).to(device)

# Dataset
class FormulaDataset(Dataset):
    def __init__(self, formulas, labels):
        enc = tokenizer(formulas, padding=True, max_length=128, truncation=True, return_tensors='pt')
        self.encodings = enc
        self.labels = torch.tensor(labels)

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.encodings.items()} | {'labels': self.labels[idx]}

    def __len__(self):
        return len(self.labels)

# Caricamento e split
df = pd.read_csv("propositional_dataset_train10k.csv")
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['formula'].tolist(), df['satisfiable'].tolist(), test_size=0.1, stratify=df['satisfiable'], random_state=42
)

train_dataset = FormulaDataset(train_texts, train_labels)
val_dataset = FormulaDataset(val_texts, val_labels)

# Metriche
def compute_metrics(p):
    preds = p.predictions.argmax(-1)
    acc = accuracy_score(p.label_ids, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(p.label_ids, preds, average='binary')
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

# Argomenti di training (adatti alla tua GPU e modello mini)
training_args = TrainingArguments(
    output_dir="./mini-bert-prop",         # Dove salvare i checkpoint
    per_device_train_batch_size=16,        # Coerente con setup CUDA
    per_device_eval_batch_size=16,
    num_train_epochs=30,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    save_total_limit=1,
    fp16=torch.cuda.is_available(),
    weight_decay= 0.01,
    learning_rate = 2e-5,
    warmup_steps = 0  # o 50

)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer
)

# Training
trainer.train()

# Valutazione finale
metrics = trainer.evaluate()
print("Final evaluation metrics:", metrics)

# Salvataggio finale
trainer.save_model("./mini-bert-prop")
tokenizer.save_pretrained("./mini-bert-prop")
