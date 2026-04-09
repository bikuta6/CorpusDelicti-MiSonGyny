import re

with open('src/task2/bert_based_comparison.py', 'r') as f:
    content = f.read()

# Replace main def
main_repl = """def main(augment=False, baseline=False):
    if baseline:
        apply_baseline_settings()
        
    pre = "processed_" if not baseline else ""
    TRAIN_PATH = f"../../data/task2/{pre}train_df.csv"
    VAL_PATH = f"../../data/task2/{pre}val_df.csv"
    DEV_PATH = f"../../data/task2/{pre}dev_df.csv"
    RESULTS_FILE = f"../../results/task2/tabla_paper_{'baseline' if baseline else 'processed'}.csv"
"""

content = re.sub(r'def main\(augment=False\):\n\s+TRAIN_PATH = DATA_PATH\n\s+VAL_PATH = DATA_PATH\.replace\("train_df\.csv", "val_df\.csv"\)\n\s+label_cols = \["sexualization", "violence", "hate"\]\n\s+print\(f"Cargando datos de entrenamiento desde \{TRAIN_PATH\}\.\.\."\)\n\s+train_df = pd\.read_csv\(TRAIN_PATH\)\n\s+train_df\["label"\] = create_label_column\(train_df, label_cols\)\n\s+print\(f"Cargando datos de validación desde \{VAL_PATH\}\.\.\."\)\n\s+val_df = pd\.read_csv\(VAL_PATH\)\n\s+val_df\["label"\] = create_label_column\(val_df, label_cols\)\n\s+DEV_PATH = DATA_PATH\.replace\("train_df\.csv", "dev_df\.csv"\)\n\s+print\(f"Cargando datos de test/dev desde \{DEV_PATH\}\.\.\."\)\n\s+dev_df = pd\.read_csv\(DEV_PATH\)\n\s+dev_df\["label"\] = create_label_column\(dev_df, label_cols\)',
main_repl + """    label_cols = ["sexualization", "violence", "hate"]
    print(f"Cargando datos de entrenamiento desde {TRAIN_PATH}...")
    train_df = pd.read_csv(TRAIN_PATH)
    train_df["label"] = create_label_column(train_df, label_cols)
    print(f"Cargando datos de validación desde {VAL_PATH}...")
    val_df = pd.read_csv(VAL_PATH)
    val_df["label"] = create_label_column(val_df, label_cols)
    print(f"Cargando datos de test/dev desde {DEV_PATH}...")
    dev_df = pd.read_csv(DEV_PATH)
    dev_df["label"] = create_label_column(dev_df, label_cols)""", content)


# Replace args
args_repl = """    arg_parser.add_argument(
        "--augment", action="store_true", help="Activar augmentación de datos"
    )
    arg_parser.add_argument(
        "--baseline", action="store_true", help="Activar baseline settings"
    )
    args = arg_parser.parse_args()
    main(augment=args.augment, baseline=args.baseline)"""

content = re.sub(r'    arg_parser\.add_argument\(\n\s+"--augment", action="store_true", help="Activar augmentación de datos"\n\s+\)\n\s+args = arg_parser\.parse_args\(\)\n\s+main\(augment=args\.augment\)', args_repl, content)

with open('src/task2/bert_based_comparison.py', 'w') as f:
    f.write(content)

