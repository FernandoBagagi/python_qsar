"""Definição e compilação das arquiteturas Baseline e ResNet50V2 (Fine-tuning)."""

import keras
from keras.applications import ResNet50V2
from keras.applications.resnet_v2 import preprocess_input
from keras.layers import (
    Activation,
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    Input,
    Lambda,
    MaxPooling2D,
    RandomRotation,
    RandomTranslation,
    RandomZoom,
    Rescaling,
)
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.regularizers import l2


def build_molecular_augmentation() -> keras.Sequential:
    """Data augmentation que preserva quiralidade (sem flip) e integridade (sem corte)."""
    return Sequential(
        [
            RandomRotation(
                factor=0.5,  # Rotação completa de -180° a +180°
                fill_mode='constant',
                fill_value=255.0,
            ),
            RandomTranslation(
                height_factor=0.08,
                width_factor=0.08,
                fill_mode='constant',
                fill_value=255.0,
            ),
            RandomZoom(
                height_factor=(
                    -0.10,
                    0.05,
                ),  # Mais zoom-out que zoom-in para evitar cortes
                width_factor=(-0.10, 0.05),
                fill_mode='constant',
                fill_value=255.0,
            ),
        ],
        name='molecular_augmentation',
    )


def build_baseline_cnn(
    input_shape: tuple[int, int, int] = (224, 224, 3),
    learning_rate: float = 1e-3,
    l2_reg: float = 1e-3,
) -> keras.Model:
    """Constrói uma CNN simples do zero com 3 blocos convolucionais."""
    inputs = Input(shape=input_shape)
    x = build_molecular_augmentation()(inputs)
    x = Rescaling(1.0 / 255.0)(x)

    for filters in (32, 64, 128):
        x = Conv2D(
            filters,
            kernel_size=(3, 3),
            padding='same',
            use_bias=False,
            kernel_regularizer=l2(l2_reg),
        )(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu', kernel_regularizer=l2(l2_reg))(x)
    x = Dropout(0.4)(x)
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs, outputs, name='Baseline_CNN')
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy'],
    )
    return model


def build_resnet50v2_model(
    input_shape: tuple[int, int, int] = (224, 224, 3),
    learning_rate: float = 5e-4,
    l2_reg: float = 1e-2,
) -> tuple[keras.Model, keras.Model]:
    """Constrói a ResNet50V2 congelada para o Estágio 1 (Feature Extraction)."""
    base_model = ResNet50V2(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape,
    )
    base_model.trainable = False

    inputs = Input(shape=input_shape)
    x = build_molecular_augmentation()(inputs)
    x = Lambda(preprocess_input, name='resnet_preprocess')(x)
    x = base_model(x, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu', kernel_regularizer=l2(l2_reg))(x)
    x = Dropout(0.5)(x)
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs, outputs, name='ResNet50V2_FineTuning')
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy'],
    )
    return model, base_model


def unfreeze_resnet_for_finetuning(
    model: keras.Model,
    base_model: keras.Model,
    fine_tune_from_layer: int = 154,
    learning_rate: float = 1e-5,
) -> None:
    """Descongela o último estágio residual (conv5_block*) para o Estágio 2."""
    base_model.trainable = True

    for layer in base_model.layers[:fine_tune_from_layer]:
        layer.trainable = False

    for layer in base_model.layers[fine_tune_from_layer:]:
        if isinstance(layer, BatchNormalization):
            layer.trainable = False

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy'],
    )
