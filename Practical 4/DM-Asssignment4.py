from tabulate import tabulate

transactions = [
    ['Laptop', 'Mouse', 'Laptop Bag'],
    ['Laptop', 'Keyboard', 'Mouse'],
    ['Mobile', 'Charger', 'Earphones'],
    ['Laptop', 'Mouse', 'Keyboard', 'Laptop Bag'],
    ['Mobile', 'Charger', 'Screen Guard'],
    ['Laptop', 'Laptop Bag', 'Mouse'],
    ['Mobile', 'Earphones', 'Screen Guard'],
    ['Laptop', 'Keyboard', 'Mouse', 'Mouse Pad'],
    ['Mobile', 'Charger', 'Earphones', 'Screen Guard'],
    ['Laptop', 'Mouse', 'Mouse Pad']
]

itemsets = [
    {'Laptop'},
    {'Mouse'},
    {'Mobile'},
    {'Laptop', 'Mouse'},
    {'Mobile', 'Charger'},
    {'Laptop', 'Mouse', 'Keyboard'}
]

rules = [
    ({'Laptop'}, {'Mouse'}),
    ({'Mobile'}, {'Charger'}),
    ({'Laptop', 'Mouse'}, {'Keyboard'}),
    ({'Charger'}, {'Mobile'})
]


def support(itemset):
    count = sum(itemset.issubset(set(t)) for t in transactions)
    return count / len(transactions)


# Q1
table1 = []

for itemset in itemsets:
    table1.append([
        str(itemset),
        round(support(itemset), 2)
    ])

print("Q1: Support of Itemsets")
print(tabulate(table1, headers=["Itemset", "Support"], tablefmt="grid"))


# Q2
table2 = []

for left, right in rules:
    rule_support = support(left | right)
    confidence = rule_support / support(left)

    table2.append([
        str(left) + " -> " + str(right),
        round(rule_support, 2),
        round(confidence, 2)
    ])

print("\nQ2: Support and Confidence of Rules")
print(tabulate(
    table2,
    headers=["Rule", "Support", "Confidence"],
    tablefmt="grid"
))