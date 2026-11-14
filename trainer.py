"""Funções de treinamento e avaliação dos modelos."""

import time

import keras
import numpy as np
from keras.callbacks import EarlyStopping

import visualization as view
from models import (
    build_baseline_cnn,
    build_resnet50v2_model,
    unfreeze_resnet_for_finetuning,
)

DatasetSplits = tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]


def _build_report_from_model(
    model: keras.Model,
    dataset_name: str,
    model_type: str,
    splits: DatasetSplits,
    history: dict[str, list[float]],
    epoch_times: list[float],
    total_time: float,
    class_names: tuple[str, str],
) -> view.EvaluationReport:
    """Calcula probabilidades nos 3 splits e instancia o EvaluationReport."""
    x_train, y_train, x_val, y_val, x_test, y_test = splits

    return view.EvaluationReport(
        dataset_name=dataset_name,
        model_type=model_type,
        history=history,
        y_train=y_train,
        y_train_prob=model.predict(x_train, verbose=0).ravel(),
        y_val=y_val,
        y_val_prob=model.predict(x_val, verbose=0).ravel(),
        y_test=y_test,
        y_test_prob=model.predict(x_test, verbose=0).ravel(),
        epoch_times=epoch_times,
        total_time=total_time,
        class_names=class_names,
    )


def train_and_evaluate_baseline(
    dataset_name: str,
    splits: DatasetSplits,
    epochs: int = 80,
    batch_size: int = 32,
    class_names: tuple[str, str] = ('Inativo', 'Ativo'),
) -> view.EvaluationReport:
    """Treina a CNN Baseline e retorna seu relatório completo de avaliação."""
    x_train, y_train, x_val, y_val, _, _ = splits

    model = build_baseline_cnn()
    timer = view.EpochTimerCallback()
    early_stop = EarlyStopping(
        monitor='val_loss', patience=20, restore_best_weights=True
    )

    hist = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop, timer],
    )

    return _build_report_from_model(
        model=model,
        dataset_name=dataset_name,
        model_type='Baseline',
        splits=splits,
        history=hist.history,
        epoch_times=timer.epoch_times,
        total_time=timer.total_time,
        class_names=class_names,
    )


def train_and_evaluate_resnet(
    dataset_name: str,
    splits: DatasetSplits,
    epochs_stage1: int = 20,
    epochs_stage2: int = 50,
    batch_size: int = 32,
    class_names: tuple[str, str] = ('Inativo', 'Ativo'),
) -> view.EvaluationReport:
    """Treina a ResNet50V2 em 2 estágios e retorna seu relatório de avaliação."""
    x_train, y_train, x_val, y_val, _, _ = splits

    model, base_model = build_resnet50v2_model(learning_rate=1e-3)

    # Estágio 1: Feature Extraction (backbone congelado)
    timer_s1 = view.EpochTimerCallback()
    es_s1 = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    hist_s1 = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs_stage1,
        batch_size=batch_size,
        callbacks=[es_s1, timer_s1],
    )

    # Estágio 2: Fine-tuning parcial do bloco conv5
    unfreeze_resnet_for_finetuning(
        model, base_model, fine_tune_from_layer=154, learning_rate=1e-5
    )
    timer_s2 = view.EpochTimerCallback()
    es_s2 = EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True)
    hist_s2 = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs_stage2,
        batch_size=batch_size,
        callbacks=[es_s2, timer_s2],
    )

    full_history = {
        k: hist_s1.history[k] + hist_s2.history[k]
        for k in ('loss', 'val_loss', 'accuracy', 'val_accuracy')
    }
    full_epoch_times = timer_s1.epoch_times + timer_s2.epoch_times
    full_total_time = timer_s1.total_time + timer_s2.total_time

    return _build_report_from_model(
        model=model,
        dataset_name=dataset_name,
        model_type='Fine-tuning (ResNet50V2)',
        splits=splits,
        history=full_history,
        epoch_times=full_epoch_times,
        total_time=full_total_time,
        class_names=class_names,
    )


def evaluate_majority_classifier(
    dataset_name: str,
    splits: DatasetSplits,
    class_names: tuple[str, str] = ('Inativo', 'Ativo'),
) -> view.EvaluationReport:
    """Avalia um classificador trivial que sempre prediz a classe majoritária do treino.
    """
    _, y_train, _, y_val, _, y_test = splits

    start_time = time.perf_counter()
    # Conta ocorrências de False (0) e True (1) no treino
    majority_class = bool(np.mean(y_train) >= 0.5)
    prob_value = 1.0 if majority_class else 0.0
    elapsed = time.perf_counter() - start_time

    return view.EvaluationReport(
        dataset_name=dataset_name,
        model_type='Dummy (Classe Majoritária)',
        history={
            'loss': [0.0],
            'val_loss': [0.0],
            'accuracy': [0.0],
            'val_accuracy': [0.0],
        },
        y_train=y_train,
        y_train_prob=np.full(len(y_train), prob_value, dtype=np.float32),
        y_val=y_val,
        y_val_prob=np.full(len(y_val), prob_value, dtype=np.float32),
        y_test=y_test,
        y_test_prob=np.full(len(y_test), prob_value, dtype=np.float32),
        epoch_times=[elapsed],
        total_time=elapsed,
        class_names=class_names,
    )
