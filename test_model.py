# test_model.py

import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, accuracy_score
import argparse
from torch.utils.data import Dataset
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def evaluate(model_path, test_csv_path, batch_size=16, output_file=None):
    print(f"Loading model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    model.eval()

    print(f"Loading test data from: {test_csv_path}")
    df = pd.read_csv(test_csv_path)
    formulas = df['formula'].tolist()
    labels = df['satisfiable'].tolist()

    dataset = FormulaDataset(formulas, labels, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    preds, golds = [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels_batch = batch['labels'].to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            pred = torch.argmax(logits, dim=1)

            preds.extend(pred.cpu().tolist())
            golds.extend(labels_batch.cpu().tolist())

    acc = accuracy_score(golds, preds)
    report = classification_report(golds, preds, digits=4)

    result_text = f"\nAccuracy: {acc:.4f}\n\nClassification Report:\n{report}"
    print(result_text)

    # Salvataggio su file
    if output_file is None:
        output_file = os.path.join(model_path, "evaluation_results.txt")
    with open(output_file, "w") as f:
        f.write(result_text)
    print(f"\nSaved results to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--test_csv_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--output_file", type=str, default=None,
                        help="Path to save evaluation results. Defaults to model_path/evaluation_results.txt")
    args = parser.parse_args()

    evaluate(args.model_path, args.test_csv_path, args.batch_size, args.output_file)
