import random
import sympy
from sympy.logic.boolalg import Or, And, Not
from sympy.abc import a, b, c, d, e, f, g
from sympy.logic.inference import satisfiable
import csv

variables = [a, b, c, d, e, f, g]

def random_formula(depth=3):
    if depth == 0:
        return random.choice(variables)

    if depth == 1:
        var = random.choice(variables)
        return Not(var) if random.random() < 0.5 else var

    op = random.choice(['AND', 'OR'])
    left = random_formula(depth - 1)
    right = random_formula(depth - 1)
    binary = And(left, right) if op == 'AND' else Or(left, right)
    return Not(binary) if random.random() < 0.5 else binary


def generate_balanced_dataset(n, depth=3):
    assert n % 2 == 0, "Il numero totale deve essere pari per bilanciare SAT e UNSAT"

    sat_formulas = []
    unsat_formulas = []

    while len(sat_formulas) < n // 2 or len(unsat_formulas) < n // 2:
        formula = random_formula(depth)
        is_sat = satisfiable(formula, all_models=False)

        if is_sat and len(sat_formulas) < n // 2:
            sat_formulas.append((str(formula), 1))
        elif not is_sat and len(unsat_formulas) < n // 2:
            unsat_formulas.append((str(formula), 0))

    dataset = sat_formulas + unsat_formulas
    random.shuffle(dataset)
    return dataset


def save_dataset_csv(dataset, path):
    with open(path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['formula', 'satisfiable'])
        writer.writerows(dataset)


def main():
    train_dataset = generate_balanced_dataset(1000, depth=3)
    test_dataset = generate_balanced_dataset(200, depth=3)

    save_dataset_csv(train_dataset, 'propositional_dataset_train.csv')
    save_dataset_csv(test_dataset, 'propositional_dataset_test.csv')

    print("Dataset salvati correttamente.")

if __name__ == "__main__":
    main()
