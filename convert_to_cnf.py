import ast
import pandas as pd
import sympy
from sympy.logic.boolalg import to_cnf
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application

transformations = (standard_transformations + (implicit_multiplication_application,))

def extract_variables(formulas):
    symbols = set()
    for f in formulas:
        for char in f:
            if char.isalpha():
                symbols.add(char)
    return sorted(symbols)

def get_symbol_table(variables):
    return {v: sympy.Symbol(v) for v in variables}

def convert_formula_to_cnf(formula_str, symbol_table):
    formula_str = formula_str.strip()
    if formula_str == "True":
        return "True"
    if formula_str == "False":
        return "False"
    try:
        expr = parse_expr(formula_str, local_dict=symbol_table, transformations=transformations)
        cnf_expr = to_cnf(expr, simplify=True)

        # Controllo se cnf_expr è un BooleanTrue o BooleanFalse
        if cnf_expr == sympy.true:
            return "True"
        elif cnf_expr == sympy.false:
            return "False"

        clauses = cnf_expr_to_list(cnf_expr)
        return str(clauses)
    except Exception as e:
        print(f"⚠️ Errore nella formula: {formula_str} -> {e}")
        return "False"  # o "ERROR" a seconda di come vuoi gestire


def cnf_expr_to_list(expr):
    """
    Converte un'espressione sympy in CNF (AND di OR) in lista di liste di int.
    Ogni lettera è mappata ad un intero positivo,
    ogni negazione ad intero negativo.
    Es: (a | ~b) & (c) -> [[1,-2],[3]]
    """
    # Raccogli simboli per creare indice numerico
    symbols = sorted(expr.atoms(sympy.Symbol), key=lambda s: s.name)
    symbol_to_int = {s: i+1 for i, s in enumerate(symbols)}

    def literal_to_int(lit):
        if isinstance(lit, sympy.Not):
            return -symbol_to_int[lit.args[0]]
        else:
            return symbol_to_int[lit]

    # Se la formula è una sola clausola (OR), o letterale singolo
    if isinstance(expr, sympy.Or):
        clause = [literal_to_int(arg) for arg in expr.args]
        return [clause]
    elif isinstance(expr, sympy.Symbol) or isinstance(expr, sympy.Not):
        return [[literal_to_int(expr)]]
    elif isinstance(expr, sympy.And):
        clauses = []
        for arg in expr.args:
            if isinstance(arg, sympy.Or):
                clause = [literal_to_int(lit) for lit in arg.args]
                clauses.append(clause)
            elif isinstance(arg, sympy.Symbol) or isinstance(arg, sympy.Not):
                clauses.append([literal_to_int(arg)])
            else:
                raise ValueError(f"Impossibile convertire la clausola: {arg}")
        return clauses
    else:
        raise ValueError(f"Impossibile convertire la formula: {expr}")

def convert_dataset_to_cnf(input_csv_path, output_csv_path):
    df = pd.read_csv(input_csv_path)
    formulas = df["formula"].tolist()

    variables = extract_variables(formulas)
    symbol_table = get_symbol_table(variables)

    cnf_formulas = []
    for f in formulas:
        cnf = convert_formula_to_cnf(f, symbol_table)
        if cnf == "ERROR":
            # In caso di errore, mantieni la formula originale o gestisci come preferisci
            cnf = "False"  # o "ERROR"
        cnf_formulas.append(cnf)

    df["cnf_formula"] = cnf_formulas
    df[["cnf_formula", "satisfiable"]].to_csv(output_csv_path, index=False)
    print(f"✔️ File salvato in: {output_csv_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    args = parser.parse_args()

    convert_dataset_to_cnf(args.input_csv, args.output_csv)
