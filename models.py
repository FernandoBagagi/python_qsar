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
    Rescaling,
)
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2


def build_baseline_cnn(
    input_shape: tuple[int, int, int] = (224, 224, 3),
    learning_rate: float = 1e-3,
    l2_reg: float = 1e-4,
) -> keras.Model:
    """Constrói uma CNN simples do zero com 3 blocos convolucionais."""
    inputs = Input(shape=input_shape)
    x = Rescaling(1.0 / 255.0)(inputs)

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
    x = Dropout(0.4)(x)
    x = Dense(64, activation='relu', kernel_regularizer=l2(l2_reg))(x)
    x = Dropout(0.3)(x)
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
    learning_rate: float = 1e-3,
) -> tuple[keras.Model, keras.Model]:
    """Constrói a ResNet50V2 congelada para o Estágio 1 (Feature Extraction)."""
    base_model = ResNet50V2(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape,
    )
    base_model.trainable = False

    inputs = Input(shape=input_shape)
    x = Lambda(preprocess_input, name='resnet_preprocess')(inputs)
    x = base_model(x, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.4)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
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
