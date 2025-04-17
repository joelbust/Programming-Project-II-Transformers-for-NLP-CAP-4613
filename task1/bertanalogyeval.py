from bertanalogysolver import load_analogies, evaluate_group


groups = load_analogies("word-test.v1.txt")

# the 3 chosen groups
selected_groups = ["capital-common-countries", "currency", "family"]

# run the evaluation for all groups
for group in selected_groups:
    acc_cos, acc_l2 = evaluate_group(group, groups[group])
    print(f"\nGroup: {group}")
    print("k\tCosine Acc\tL2 Acc")
    for k in [1, 2, 5, 10, 20]:
        print(f"{k}\t{acc_cos[k]:.3f}\t\t{acc_l2[k]:.3f}")
