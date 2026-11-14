"""Funções responsáveis pelo carregamento e pré-processamento do dataset."""

from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


def load_dataset(
    dataset_name: str,
    datasets_dir: str = 'datasets',
    shuffle_train: bool = True,
    deterministic: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Carrega e prepara as partições (train, val e test) de um dataset de imagens.

    Lê as imagens PNG das subpastas ``class_0`` e ``class_1`` de cada split,
    converte para RGB, redimensiona para ``(224, 224)`` pixels e concatena
    as classes.

    Os rótulos são representados como valores booleanos:
        - ``False`` -> ``class_0``
        - ``True`` -> ``class_1``

    Parameters
    ----------
    dataset_name : str
        Nome da pasta do dataset dentro do diretório base.
    datasets_dir : str, default='datasets'
        Nome do diretório raiz onde os datasets estão armazenados.
    shuffle_train : bool, default=True
        Se ``True``, embaralha as amostras e rótulos apenas do split de treino
        (``train``) mantendo o alinhamento entre eles. Os splits ``val`` e
        ``test`` permanecem na ordem original.
    deterministic : bool, default=False
        Se ``True``, utiliza uma semente fixa (``seed=42``) no gerador
        aleatório para garantir reprodutibilidade quando ``shuffle_train=True``.
        Se ``False``, utiliza uma semente aleatória do sistema.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Tupla contendo ``(x_train, y_train, x_val, y_val, x_test, y_test)``, onde:
        - ``x_*`` : ``np.ndarray`` com formato ``(N, 224, 224, 3)`` e dtype ``uint8``.
        - ``y_*`` : ``np.ndarray`` com formato ``(N,)`` e dtype ``bool_``.
    """
    dataset_path = Path(datasets_dir, dataset_name)
    dataset: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for split_name in ('train', 'val', 'test'):
        split_path = dataset_path / split_name

        x_0, y_0 = load_class_images(split_path, is_class_1=False)
        x_1, y_1 = load_class_images(split_path, is_class_1=True)

        x = np.concatenate([x_0, x_1], axis=0)
        y = np.concatenate([y_0, y_1], axis=0)

        if shuffle_train and split_name == 'train':
            seed = 42 if deterministic else None
            rng = np.random.default_rng(seed=seed)
            indices = rng.permutation(len(x))
            x, y = x[indices], y[indices]

        dataset[split_name] = (x, y)

    x_train, y_train = dataset['train']
    x_val, y_val = dataset['val']
    x_test, y_test = dataset['test']

    return x_train, y_train, x_val, y_val, x_test, y_test


def load_class_images(
    split_path: Path,
    is_class_1: bool,
    image_size: tuple[int, int] = (224, 224),
) -> tuple[np.ndarray, np.ndarray]:
    """Carrega, redimensiona e rotula todas as imagens PNG de uma classe específica.

    Lê as imagens do subdiretório ``class_1`` (se ``is_class_1=True``) ou ``class_0``
    (se ``is_class_1=False``), converte para o espaço de cores RGB e redimensiona
    para as dimensões especificadas.

    Parameters
    ----------
    split_path : Path
        Caminho base da partição (ex: ``Path('datasets/dataset_name/train')``).
    is_class_1 : bool
        Define qual pasta ler e o valor dos rótulos retornados.
        ``True`` lê ``class_1``; ``False`` lê ``class_0``.
    image_size : tuple[int, int], default=(224, 224)
        Tupla ``(largura, altura)`` para redimensionamento.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Uma tupla ``(images, labels)`` contendo:
        - ``images`` : Array NumPy ``uint8`` de shape ``(N, altura, largura, 3)``.
        - ``labels`` : Array NumPy booleano (``bool_``) de shape ``(N,)`` preenchido
          com o valor de ``is_class_1``.
    """

    class_dir_name = 'class_1' if is_class_1 else 'class_0'
    class_path = split_path / class_dir_name
    image_paths = sorted(class_path.glob('*.png'))

    images = []
    description = f'Carregando {split_path.name}/{class_dir_name}'
    for image_path in tqdm(image_paths, desc=description, unit='img'):
        with Image.open(image_path) as image:
            image_rgb = image.convert('RGB')
        image_resized = image_rgb.resize(image_size)
        images.append(image_resized)

    images_array = np.asarray(images, dtype=np.uint8)
    labels_array = np.full(len(images_array), is_class_1, dtype=np.bool_)

    return images_array, labels_array
