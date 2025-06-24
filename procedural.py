import pandas as pd
import ast
from solver import Solver  # Assumendo che il tuo codice Solver sia salvato in solver.py

def interpret_formula(formula_str):
    formula_str = formula_str.strip()
    if formula_str == "False":
        return False
    elif formula_str == "True":
        return True
    else:
        try:
            return ast.literal_eval(formula_str)
        except Exception as e:
            print(f" Errore parsing formula: {formula_str} -> {e}")
            return None

def evaluate_dataset(csv_path):
    df = pd.read_csv(csv_path)
    total = 0
    correct = 0

    for i, row in df.iterrows():
        formula_str = str(row['cnf_formula'])
        expected = int(row['satisfiable'])

        parsed = interpret_formula(formula_str)

        if parsed is True:
            result = True
        elif parsed is False:
            result = False
        elif isinstance(parsed, list):
            try:
                solver = Solver(parsed)
                result = solver.solve()
            except Exception as e:
                print(f" Errore nella risoluzione formula alla riga {i}: {e}")
                continue
        else:
            print(f" Formula non riconosciuta alla riga {i}: {formula_str}")
            continue

        if int(result) == expected:
            correct += 1
        else:
            print(f" Risultato errato alla riga {i}: formula={formula_str}, expected={expected}, got={int(result)}")
        total += 1

    accuracy = correct / total * 100 if total > 0 else 0
    print(f"\n Accuracy finale: {accuracy:.2f}% su {total} esempi.")

if __name__ == "__main__":
    evaluate_dataset("cnf_dataset_test.csv")
