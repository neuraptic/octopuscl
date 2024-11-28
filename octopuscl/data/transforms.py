""" Module for data transformations. """

from abc import ABC
from abc import abstractmethod
import itertools
import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from octopuscl.types import Example
from octopuscl.types import PipelineMode


class Transform(ABC):
    """
    Base class for data transformations.
    """

    def __init__(self, mode: PipelineMode, *_args, **_kwargs):
        """
        Default constructor.

        Args:
            mode (PipelineMode): Mode in which the transformation will be applied (training or evaluation).
        """
        self._mode = mode

    @property
    def mode(self) -> PipelineMode:
        return self._mode

    @mode.setter
    def mode(self, mode: PipelineMode):
        self._mode = mode

    @abstractmethod
    def transform(self, example: Example) -> Example:
        """
        Applies the transformation to the provided example.

        Args:
            example (Example): Example to apply the transformation to.

        Returns:
            Example: The transformed example.
        """
        raise NotImplementedError()

    def __call__(self, example: Example) -> Example:
        """
        Applies the transformation to the provided example, calling `self.transform()`.

        Args:
            example (Example): Example to apply the transformation to.

        Returns:
            Example: Transformed example.
        """
        return self.transform(example)


class TransformEstimator(Transform):
    """ Abstract class for transformations that require fitting. """

    # TODO: Any way to enforce that `fit` is called before `transform` in any subclass?

    @abstractmethod
    def fit(self, examples: Iterable[Example]):
        """
        Fits the transformation on the input dataset.

        Args:
            examples (Iterable[Example]): The examples to fit the transformation on.
        """
        raise NotImplementedError()


TrainEvalTransforms = Tuple[Optional['TransformChain'], Optional['TransformChain']]


class TransformChain:
    """ Data transformations to be applied sequentially. """

    def __init__(self, transforms: Sequence[Transform]):
        """
        Default constructor.

        Args:
            transforms (Sequence[Transform]): Transformations to be applied sequentially.

        Raises:
            ValueError: If `transforms` is empty or `None`.
        """
        if not transforms:
            raise ValueError("'transforms' must not be empty")
        self._transforms = list(transforms)

    @property
    def transforms(self) -> List[Transform]:
        return list(self._transforms)  # Return a copy of the list to avoid external modifications

    @classmethod
    def init_training_and_evaluation_transforms(cls, transforms_configs: List[dict]) -> TrainEvalTransforms:
        """
        Creates a `TransformChain` object for training and evaluation from a list of transformation configurations.

        Args:
            transforms_configs (List[dict]): List of dictionaries where each dictionary represents a transformation
                                             configuration.

        Returns:
            TrainEvalTransforms: A tuple of `TransformChain` objects with all the transformations to be applied in
                                 training and evaluation, respectively.

        Example:
            Given a list of transformation configurations like this:
            [
                {
                    "class": "my_module.MyTransformationClass1",
                    "mode": ["train", "eval"],
                    "parameters": {"param1": "value1"}
                },
                {
                    "class": "my_module.MyTransformationClass2",
                    "mode": ["train"]
                }
            ]

            Calling `init_training_and_evaluation_transforms` will produce a tuple of two objects:
                1. `TransformChain` with the two transformations specified for training.
                2. `TransformChain` with the transformation specified for evaluation.
        """

        transforms = {PipelineMode.TRAIN: [], PipelineMode.EVAL: []}

        for cfg in transforms_configs:
            tfm_type = cfg['class_']
            assert issubclass(tfm_type, Transform)
            params = cfg.get('parameters', dict())
            modes = cfg['mode']
            for mode in (modes if isinstance(modes, list) else [modes]):
                tfm = tfm_type(mode=mode, **params)
                transforms[mode].append(tfm)

        train_transforms_list = transforms[PipelineMode.TRAIN]
        eval_transforms_list = transforms[PipelineMode.EVAL]

        train_transforms = TransformChain(transforms=train_transforms_list) if train_transforms_list else None
        eval_transforms = TransformChain(transforms=eval_transforms_list) if eval_transforms_list else None

        return train_transforms, eval_transforms

    def fit(self, examples: Iterable[Example]):
        """
        Fits the transformations on the provided examples.

        Args:
            examples (Iterable[Example]): The examples to fit the transformation on.
        """

        # TODO: This function may not be efficient for large datasets.

        def apply_transforms(original_examples: Iterable[Example],
                             fitted_transforms_: List[Transform]) -> Iterable[Example]:
            """
            Applies the transformations fitted so far to the original (untransformed) examples.

            Args:
                original_examples (Iterable[Example]): Original (untransformed) examples.
                fitted_transforms_ (List[Transform]): Fitted transformations to be applied to each example.

            Yields:
                Iterable[Example]: Examples transformed by the transformations fitted so far.
            """
            for v in original_examples:
                for f in fitted_transforms_:
                    v = f(v)
                yield v

        # Keeps track of fitted transformations
        fitted_transforms = []

        # Sequentially fit transformations based on the examples transformed by the previously fitted transformations
        for tfm in self.transforms:
            # Ensure examples can be re-iterated
            examples, examples_to_fit = itertools.tee(examples)
            # Fit the transformation. Two cases.
            #     1. First transformation: Fit it on the original examples.
            #     2. Subsequent transformations: Fit it on the examples
            #        transformed by the previously fitted transformations.
            if fitted_transforms:
                # Apply the transformations fitted so far
                transformed_examples = apply_transforms(examples_to_fit, fitted_transforms)
                # Fit the transformation on the transformed examples
                if isinstance(tfm, TransformEstimator):
                    tfm.fit(transformed_examples)
            else:
                # Fit the transformation on the original examples
                if isinstance(tfm, TransformEstimator):
                    tfm.fit(examples_to_fit)
            # Track fitted transformation
            fitted_transforms.append(tfm)

    def transform(self, example: Example) -> Example:
        """
        Sequentially applies all the transformations to the provided example.

        Args:
            example (Example): Example to apply the transformation to

        Returns:
            Example: Transformed example
        """
        for tfm in self.transforms:
            example = tfm(example)
        return example

    def __call__(self, example: Example) -> Example:
        """
        Sequentially applies all the transformations to the provided example.

        Args:
            example (Example): Example to apply the transformation to.

        Returns:
            Example: Transformed example.
        """
        return self.transform(example)

    def set_mode(self, mode: PipelineMode):
        for tfm in self.transforms:
            tfm.mode = mode


class FeatureSelection(Transform):
    """ Feature selection. """

    def __init__(self, mode: PipelineMode, features: Iterable[str]):
        """
        Default constructor.

        Args:
            mode (PipelineMode): Mode in which the transformation will be applied (training or evaluation).
            features (Iterable[str]): Name of the features to select.
        """
        super().__init__(mode=mode)
        self._features = list(features)

    @property
    def features(self) -> List[str]:
        return list(self._features)  # Return a copy of the list to avoid external modifications

    def transform(self, example: Example) -> Example:
        """
        Applies the transformation to the provided example.

        Args:
            example (Example): Example to apply the transformation to.

        Returns:
            Example: The transformed example.
        """
        return {k: example[k] for k in self.features}


class StandardScaler(TransformEstimator):
    """ Standard scaler transformation. """

    # TODO: This is not filtering the features to be transformed, right? Check `self.features`

    def __init__(self, mode: PipelineMode, features: Iterable[str]):
        """
        Default constructor.

        Args:
            mode (PipelineMode): Mode in which the transformation will be applied (training or evaluation).
            features (Iterable[str]): Name of the features to apply the transformation to.
        """
        super().__init__(mode=mode)
        self._features = list(features)
        self._stats: Dict[str, Optional[Tuple[float, float]]] = {k: None for k in features}

    @property
    def features(self) -> List[str]:
        return list(self._features)  # Return a copy of the list to avoid external modifications

    def fit(self, examples: Iterable[Example]):
        """
        Computes the mean and standard deviation for each feature.

        Args:
            examples (Iterable[Example]): The examples to fit the transformation on.
        """
        # Duplicate iterator: use one to compute mean and the other one to compute std
        x1, x2 = itertools.tee(examples, 2)
        # Initialize mean stats (sum and counter) with 0s
        running_stats = {k: [0, 0] for k in self.features}
        # For each example
        for v in x1:
            # For each feature
            for k in self.features:
                if k not in v:
                    raise ValueError(f"Missing '{k}' feature in the dataset")
                # Sum value and counter
                running_stats[k][0] += v[k]
                running_stats[k][1] += 1

        # Compute mean. When counter is 0 (no elements) set mean to 0 so it has no effect when scaling
        mean_stat = {
            k: running_stats[k][0] / running_stats[k][1] if running_stats[k][1] != 0 else 0 for k in self.features
        }

        # Initialize std stats (sum and counter) with 0s
        running_stats = {k: [0, 0] for k in self.features}
        # For each example
        for v in x2:
            # For each feature
            for k in self.features:
                if k not in v:
                    raise ValueError(f"Missing '{k}' feature in the dataset")
                # Compute example std and sum 1 to counter
                running_stats[k][0] += (v[k] - mean_stat[k])**2
                running_stats[k][1] += 1

        # Compute std. When counter is 0 (no elements) set std to 0 so it has no effect when scaling
        std_stat = {k: math.sqrt(running_stats[k][0] / running_stats[k][1]) for k in self.features}

        self._stats = {k: (mean_stat[k], std_stat[k]) for k in self.features}

    def transform(self, example: Example) -> Example:
        """
        Applies the transformation to the provided example.

        Args:
            example (Example): Example to apply the transformation to.

        Returns:
            Example: The transformed example.
        """
        return {k: (v - self._stats[k][0]) / self._stats[k][1] if k in self._stats else v for k, v in example.items()}


class OutputEncoder(TransformEstimator):
    """ Encodes output values. """

    def __init__(self, mode: PipelineMode, outputs: Iterable[str]):
        """
        Default constructor.

        Args:
            mode (PipelineMode): Mode in which the transformation will be applied (training or evaluation).
            outputs (Iterable[str]): Name of the outputs to apply the transformation to.
        """
        super().__init__(mode=mode)
        self._outputs = list(outputs)
        self._code_book = {k: {} for k in outputs}

    @property
    def outputs(self) -> List[str]:
        return list(self._outputs)  # Return a copy of the list to avoid external modifications

    @property
    def code_book(self) -> (Dict[str, Dict[Any, int]]):
        """ Returns the codification for each output. """
        return self._code_book

    def fit(self, examples: Iterable[Example]):
        """
        Extracts the codification for each output.

        Args:
            examples (Iterable[Example]): The examples to fit the transformation on.
        """
        cur_code = {k: 0 for k in self.outputs}
        for i in examples:
            for k in self.outputs:
                v = i[k]
                if v not in self.code_book[k]:
                    self.code_book[k][v] = cur_code[k]
                    cur_code[k] += 1

    def transform(self, example: Example) -> Example:
        """
        Applies the transformation to the provided example.

        Args:
            example (Example): Example to apply the transformation to.

        Returns:
            Example: The transformed example.
        """
        return {k: self.code_book[k][v] if k in self.outputs else v for k, v in example.items()}
