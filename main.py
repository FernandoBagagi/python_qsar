"""Script principal de execução dos experimentos QSAR."""

import keras
import tensorflow as tf

import visualization as view
from data_loader import load_dataset
from trainer import (
    evaluate_majority_classifier,
    train_and_evaluate_baseline,
    train_and_evaluate_resnet,
)

# ========================
# Configuração dos experimentos
# ========================

keras.utils.set_random_seed(42)
tf.config.experimental.enable_op_determinism()

DATASETS = (
    'bace_c',
    'bbbp',
    'toxicidade',
)
all_reports: list[view.EvaluationReport] = []

for dataset_name in DATASETS:
    # ========================
    # Carregamento do dataset
    # ========================

    splits = load_dataset(dataset_name, deterministic=True)
    x_train, y_train, x_val, y_val, x_test, y_test = splits

    # ========================
    # Tamanhos dataset/split
    # ========================

    view.show_split_infos('train', x_train, y_train)
    view.show_split_infos('val', x_val, y_val)
    view.show_split_infos('test', x_test, y_test)

    # ========================
    # Baseline Trivial (Classe Majoritária)
    # ========================

    report_dummy = evaluate_majority_classifier(dataset_name, splits)
    report_dummy.print_summary()
    all_reports.append(report_dummy)

    # ========================
    # Treinamento e avaliação — Baseline
    # ========================

    report_baseline = train_and_evaluate_baseline(dataset_name, splits)
    report_baseline.print_summary(save_path=f'resumo_{dataset_name}_baseline.txt')
    report_baseline.plot_dashboard(save_path=f'dashboard_{dataset_name}_baseline.png',show=False)
    all_reports.append(report_baseline)

    # ========================
    # Treinamento e avaliação — Fine-tuning
    # ========================

    report_resnet = train_and_evaluate_resnet(dataset_name, splits)
    report_resnet.print_summary(save_path=f'resumo_{dataset_name}_resnet50v2.txt')
    report_resnet.plot_dashboard(save_path=f'dashboard_{dataset_name}_resnet50v2.png',show=False)
    all_reports.append(report_resnet)

# ========================
# Tabela comparativa final
# ========================

df_comparison = view.build_comparison_table(all_reports)
print(df_comparison.to_markdown(index=False))
view.save_experiment_report(all_reports, save_path='resultados_comparativos.txt')
