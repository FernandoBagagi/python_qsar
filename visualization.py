"""Módulo de métricas, relatórios de avaliação e visualização de resultados."""

import time
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.callbacks import Callback
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
)


def show_split_infos(split_name: str, x: np.ndarray, y: np.ndarray) -> None:

    print(f'{split_name.capitalize()}:')

    print(f'x_{split_name}: {x.shape} -> {x.dtype}')

    print(f'y_{split_name}: {y.shape} -> {y.dtype} {np.unique(y)}\n')


class EpochTimerCallback(Callback):
    """Callback customizado para registrar o tempo por época e o tempo total.

    Compatível com a interface de callbacks do Keras/TensorFlow
    (`tf.keras.callbacks.Callback`). Caso herde explicitamente de
    `tf.keras.callbacks.Callback`, basta adicionar a herança.
    """

    def __init__(self) -> None:
        self.epoch_times: list[float] = []
        self._epoch_start: float = 0.0
        self._train_start: float = 0.0
        self.total_time: float = 0.0

    def on_train_begin(self, logs: dict[str, Any] | None = None) -> None:
        _ = logs
        self.epoch_times = []
        self._train_start = time.perf_counter()

    def on_epoch_begin(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        _, _ = epoch, logs
        self._epoch_start = time.perf_counter()

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        _, _ = epoch, logs
        elapsed = time.perf_counter() - self._epoch_start
        self.epoch_times.append(elapsed)

    def on_train_end(self, logs: dict[str, Any] | None = None) -> None:
        _ = logs
        self.total_time = time.perf_counter() - self._train_start


@dataclass
class EvaluationReport:
    """Armazena dados numéricos, métricas e gera visualizações de um experimento.

    Attributes
    ----------
    dataset_name : str
        Nome do dataset avaliado.
    model_type : str
        Identificador da abordagem (ex: ``'Baseline'`` ou ``'Fine-tuning'``).
    history : dict[str, list[float]]
        Dicionário contendo ``'loss'``, ``'val_loss'``, ``'accuracy'``
        e ``'val_accuracy'``.
    y_train : np.ndarray
        Rótulos verdadeiros do conjunto de treino.
    y_train_prob : np.ndarray
        Probabilidades preditas para a classe positiva no conjunto de treino.
    y_val : np.ndarray
        Rótulos verdadeiros do conjunto de validação.
    y_val_prob : np.ndarray
        Probabilidades preditas para a classe positiva no conjunto de validação.
    y_test : np.ndarray
        Rótulos verdadeiros do conjunto de teste.
    y_test_prob : np.ndarray
        Probabilidades preditas para a classe positiva no conjunto de teste.
    epoch_times : list[float]
        Lista com a duração (em segundos) de cada época executada.
    total_time : float
        Tempo total de treinamento até a convergência/early stopping (em segundos).
    threshold : float, default=0.5
        Limiar de decisão para converter probabilidades em classes binárias.
    class_names : tuple[str, str], default=('Classe 0', 'Classe 1')
        Rótulos textuais das classes para exibição na matriz de confusão.
    """

    dataset_name: str
    model_type: str
    history: dict[str, list[float]]
    y_train: np.ndarray
    y_train_prob: np.ndarray
    y_val: np.ndarray
    y_val_prob: np.ndarray
    y_test: np.ndarray
    y_test_prob: np.ndarray
    epoch_times: list[float]
    total_time: float
    threshold: float = 0.5
    class_names: tuple[str, str] = ('Classe 0', 'Classe 1')

    # --- 6.1.1 Acurácias ---
    @property
    def train_accuracy(self) -> float:
        y_pred = self.y_train_prob >= self.threshold
        return float(accuracy_score(self.y_train, y_pred))

    @property
    def val_accuracy(self) -> float:
        y_pred = self.y_val_prob >= self.threshold
        return float(accuracy_score(self.y_val, y_pred))

    @property
    def test_accuracy(self) -> float:
        y_pred = self.y_test_prob >= self.threshold
        return float(accuracy_score(self.y_test, y_pred))

    # --- 6.1.2 AUC-ROC (Teste) ---
    @property
    def test_roc_auc(self) -> float:
        return float(roc_auc_score(self.y_test, self.y_test_prob))

    # --- 6.2.3 Matriz de Confusão (Teste) ---
    @property
    def y_test_pred(self) -> np.ndarray:
        return self.y_test_prob >= self.threshold

    @property
    def test_confusion_matrix(self) -> np.ndarray:
        """Retorna matriz 2x2 no formato [[TN, FP], [FN, TP]]."""
        return confusion_matrix(self.y_test, self.y_test_pred)

    @property
    def confusion_breakdown(self) -> dict[str, int]:
        """Decompõe a matriz de confusão em TN, FP, FN e TP para análise."""
        tn, fp, fn, tp = self.test_confusion_matrix.ravel()
        return {'TN': int(tn), 'FP': int(fp), 'FN': int(fn), 'TP': int(tp)}

    # --- 6.2.1 Melhor Época ---
    @property
    def best_epoch(self) -> int:
        """Retorna a melhor época (baseada em 1) pelo menor ``val_loss``."""
        if 'val_loss' in self.history and self.history['val_loss']:
            return int(np.argmin(self.history['val_loss'])) + 1
        return len(self.history['loss'])

    # --- 6.3.1 Tempo Médio por Época ---
    @property
    def mean_epoch_time(self) -> float:
        return float(np.mean(self.epoch_times)) if self.epoch_times else 0.0

    # --- 6.4 Linha para Tabela Comparativa ---
    def to_summary_dict(self) -> dict[str, Any]:
        """Exporta as métricas consolidadas para composição da tabela final."""
        return {
            'Dataset': self.dataset_name,
            'Modelo': self.model_type,
            'Acurácia Treino': round(self.train_accuracy, 4),
            'Acurácia Validação': round(self.val_accuracy, 4),
            'Acurácia Teste': round(self.test_accuracy, 4),
            'AUC-ROC (Teste)': round(self.test_roc_auc, 4),
            'Tempo médio por época (s)': round(self.mean_epoch_time, 2),
            'Tempo total (s)': round(self.total_time, 2),
            'Melhor Época': self.best_epoch,
            'Épocas Executadas': len(self.history['loss']),
        }

    def print_summary(self) -> None:
        """Exibe no console o resumo numérico e a decomposição FP/FN."""
        cm = self.confusion_breakdown
        print(f'=== {self.dataset_name} | {self.model_type} ===')
        print(f'Acurácia Treino:    {self.train_accuracy:.4f}')
        print(f'Acurácia Validação: {self.val_accuracy:.4f}')
        print(f'Acurácia Teste:     {self.test_accuracy:.4f}')
        print(f'AUC-ROC (Teste):    {self.test_roc_auc:.4f}')
        print(
            f'Matriz Confusão:    TN={cm["TN"]}, FP={cm["FP"]}, '
            f'FN={cm["FN"]}, TP={cm["TP"]}'
        )
        print(
            f'Tempo Médio/Época:  {self.mean_epoch_time:.2f}s '
            f'(Total: {self.total_time:.2f}s | Melhor Época: {self.best_epoch})\n'
        )

    # --- Janela Única com Todos os Gráficos Solicitados ---
    def plot_dashboard(self, save_path: str | None = None) -> None:
        """Renderiza uma única janela 2x2 com curvas de aprendizado, matriz de confusão
        e ROC."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 11))
        fig.suptitle(
            f'Relatório de Avaliação — {self.dataset_name} ({self.model_type})',
            fontsize=14,
            fontweight='bold',
        )

        epochs = range(1, len(self.history['loss']) + 1)
        best_ep = self.best_epoch

        # 1. Curva de Loss (6.2.1)
        ax_loss = axes[0, 0]
        ax_loss.plot(
            epochs, self.history['loss'], 'r-o', markersize=4, label='Train Loss'
        )
        if self.history.get('val_loss'):
            ax_loss.plot(
                epochs,
                self.history['val_loss'],
                'b-o',
                markersize=4,
                label='Validation Loss',
            )
        ax_loss.axvline(
            x=best_ep, color='g', linestyle='--', label=f'Melhor Época ({best_ep})'
        )
        ax_loss.set_title('Evolução da Loss por Época')
        ax_loss.set_xlabel('Épocas')
        ax_loss.set_ylabel('Loss')
        ax_loss.legend()
        ax_loss.grid(True, alpha=0.3)

        # 2. Curva de Acurácia (6.2.1)
        ax_acc = axes[0, 1]
        ax_acc.plot(
            epochs,
            self.history['accuracy'],
            'r-o',
            markersize=4,
            label='Train Accuracy',
        )
        if self.history.get('val_accuracy'):
            ax_acc.plot(
                epochs,
                self.history['val_accuracy'],
                'b-o',
                markersize=4,
                label='Validation Accuracy',
            )
        ax_acc.axvline(
            x=best_ep, color='g', linestyle='--', label=f'Melhor Época ({best_ep})'
        )
        ax_acc.set_title('Evolução da Acurácia por Época')
        ax_acc.set_xlabel('Épocas')
        ax_acc.set_ylabel('Acurácia')
        ax_acc.legend()
        ax_acc.grid(True, alpha=0.3)

        # 3. Matriz de Confusão no Teste (6.2.3)
        ax_cm = axes[1, 0]
        ConfusionMatrixDisplay.from_predictions(
            self.y_test,
            self.y_test_pred,
            display_labels=self.class_names,
            cmap='Blues',
            ax=ax_cm,
            colorbar=False,
        )
        cm = self.confusion_breakdown
        ax_cm.set_title(f'Matriz de Confusão (Teste)\nFP: {cm["FP"]} | FN: {cm["FN"]}')

        # 4. Curva ROC no Teste (6.1.2)
        ax_roc = axes[1, 1]
        RocCurveDisplay.from_predictions(
            self.y_test,
            self.y_test_prob,
            name=f'{self.model_type} (AUC = {self.test_roc_auc:.4f})',
            ax=ax_roc,
        )
        ax_roc.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Aleatório (AUC = 0.5000)')
        ax_roc.set_title('Curva ROC — Conjunto de Teste')
        ax_roc.legend(loc='lower right')
        ax_roc.grid(True, alpha=0.3)

        plt.tight_layout(rect=(0, 0, 1, 0.96))
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def build_comparison_table(reports: list[EvaluationReport]) -> pd.DataFrame:
    """Consolida uma lista de relatórios na tabela comparativa final (Seção 6.4)."""
    rows = [report.to_summary_dict() for report in reports]
    return pd.DataFrame(rows)
