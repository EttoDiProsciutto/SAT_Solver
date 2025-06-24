import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import Dataset
import pandas as pd
import os

# Controllo GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Cartella salvataggio
SAVE_DIR = "./trained_sat_unsat_model"
os.makedirs(SAVE_DIR, exist_ok=True)

# Dataset personalizzato
class FormulaDataset(Dataset):
    def __init__(self, formulas, labels, tokenizer):
        self.encodings = tokenizer(formulas, padding=True, truncation=True, return_tensors='pt')
        self.labels = torch.tensor(labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)

# Funzione metriche
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    prec, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    return {
        'accuracy': acc,
        'precision': prec,
        'recall': recall,
        'f1': f1
    }

def main():
    # Caricamento CSV
    df = pd.read_csv('propositional_dataset_train.csv')
    formulas = df['formula'].tolist()
    labels = df['satisfiable'].tolist()

    # Suddivisione train/val (es. 10% validazione)
    formulas_train, formulas_val, labels_train, labels_val = train_test_split(
        formulas, labels, test_size=0.1, random_state=42, stratify=labels)

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2).to(device)

    train_dataset = FormulaDataset(formulas_train, labels_train, tokenizer)
    val_dataset = FormulaDataset(formulas_val, labels_val, tokenizer)

    training_args = TrainingArguments(
        output_dir=SAVE_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        learning_rate=3e-5,
        logging_dir='./logs',
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model='accuracy',
        greater_is_better=True,
        fp16=torch.cuda.is_available()
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    print("Start fine-tuning...")
    trainer.train()

    print("Final evaluation on validation set:")
    print(trainer.evaluate())

    print("Saving model and tokenizer...")
    model.save_pretrained(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)
    print(f"Model saved in {SAVE_DIR}")

if __name__ == "__main__":
    main()
