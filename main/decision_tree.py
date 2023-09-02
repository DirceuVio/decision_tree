from typing import List, Union, Any, Dict, Tuple
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod


class Node:
    """
    Represents a node in a decision tree.

    Parameters
    ----------
    left_node : Node
        The left child node.
    right_node : Node
        The right child node.
    split_metric : Union[int, str]
        The index or name of the feature used for splitting.
    split_metric_index : int
        The index of the feature used for splitting.
    split_value : float
        The value of the feature used for splitting.
    """

    def __init__(
        self,
        left_node,
        right_node,
        split_metric: Union[int, str],
        split_metric_index: int,
        split_value: float,
    ) -> None:
        self.left_node = left_node
        self.right_node = right_node
        self.split_metric = split_metric
        self.split_metric_index = split_metric_index
        self.split_value = split_value

    def __repr__(self) -> str:
        """
        Return a string representation of the Node.

        Returns
        -------
        str
            A formatted string describing the Node.
        """
        return f"{self.split_metric} <= {self.split_value} -> Yes {self.left_node} -> No {self.right_node}"


class BaseTree(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def predict(self):
        pass

    def _get_potential_splits(
        self, X: Union[np.ndarray, pd.DataFrame]
    ) -> Dict[int, List[Any]]:
        """
        Get potential split values for all variables.

        Parameters
        ----------
        X : Union[np.ndarray, pd.DataFrame]
            The feature matrix.

        Returns
        -------
        Dict[int, List[Any]]
            A dictionary containing potential split values for each column.
        """
        potential_splits = {}
        X = np.array(X)
        _, cols = X.shape

        for col in range(cols):
            unique_values = np.unique(X[:, col])
            try:
                potential_splits[col] = []
                unique_values = unique_values.astype(np.float64)
                for index in range(len(unique_values)):
                    if index != 0:
                        split_value = (
                            unique_values[index - 1] + unique_values[index]
                        ) / 2
                        potential_splits[col].append(split_value)
            except:
                potential_splits[col] = list(unique_values)

        return potential_splits

    def _split_data(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        split_column: int,
        split_value: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split the data into two parts based on the split value.

        Parameters
        ----------
        X : Union[np.ndarray, pd.DataFrame]
            The feature matrix.
        y : Union[np.ndarray, pd.Series]
            The target variable.
        split_column : int
            The index of the column to split on.
        split_value : float
            The value to split on.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            Four arrays representing the split data.
        """
        X = np.array(X)
        y = np.array(y)

        data = np.column_stack((X, y))

        split_column_values = data[:, split_column]

        data1, data2 = (
            data[split_column_values <= split_value],
            data[split_column_values > split_value],
        )

        X1, y1 = data1[:, :-1], data1[:, -1]
        X2, y2 = data2[:, :-1], data2[:, -1]

        return (X1, y1, X2, y2)


class DecisionTreeClassifier(BaseTree):
    """
    Decision Tree Classifier Algorithm
    """

    def __init__(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        min_samples_leaf: Union[int, None] = None,
        max_depth: Union[int, None] = None,
    ) -> None:
        """
        Initialize a DecisionTreeClassifier object.

        Parameters
        ----------
        X : Union[np.ndarray, pd.DataFrame]
            The feature matrix.
        y : Union[np.ndarray, pd.Series]
            The target variable.
        """
        self.X = X
        self.y = y
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth

        try:
            self.feature_names = self.X.columns
        except:
            self.feature_names = None

    def fit(self) -> None:
        """
        Fit the decision tree on the provided data.
        """
        self._fitted_tree = self._fit_tree(self.X, self.y)

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Predict the class labels for the input data.

        Parameters
        ----------
        X : Union[np.ndarray, pd.DataFrame]
            The feature matrix for prediction.

        Returns
        -------
        np.ndarray
            The predicted class labels.
        """
        X = np.array(X)
        predictions = []

        for line in X:
            predictions.append(self._predict_line(line))

        return np.array(predictions)

    def _predict_line(
        self, X: Union[np.ndarray, pd.DataFrame], tree: Union[Node, None] = None
    ) -> Any:
        """
        Predict the class label for a single input instance.

        Parameters
        ----------
        X : Union[np.ndarray, pd.DataFrame]
            The feature matrix for prediction.
        tree : Union[Node, None], optional
            The decision tree node to start prediction from, by default None.

        Returns
        -------
        Any
            The predicted class label.
        """
        if tree is None:
            tree = self._fitted_tree

        if not isinstance(tree, Node):
            return tree

        if X[tree.split_metric_index].item() <= tree.split_value:
            return self._predict_line(X, tree.left_node)
        else:
            return self._predict_line(X, tree.right_node)

    def _fit_tree(
        self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]
    ) -> Union[Node, Any]:
        """
        Recursively build the decision tree.

        Parameters
        ----------
        X : Union[np.ndarray, pd.DataFrame]
            The feature matrix.
        y : Union[np.ndarray, pd.Series]
            The target variable.

        Returns
        -------
        Union[Node, Any]
            The root node of the built decision tree or predicted value for a leaf.
        """
        depth = 0

        if self.max_depth is None:
            self.max_depth = np.inf

        if self.min_samples_leaf is None:
            self.min_samples_leaf = 0

        if (
            self._check_purity(y)
            or len(X) <= self.min_samples_leaf
            or depth >= self.max_depth
        ):
            classification = self._classify_data(y)
            return classification
        else:
            depth += 1
            potential_splits = self._get_potential_splits(X)
            best_split_column, best_split_value = self._best_split(
                X, y, potential_splits
            )
            X1, y1, X2, y2 = self._split_data(X, y, best_split_column, best_split_value)

        left_tree = self._fit_tree(X1, y1)
        right_tree = self._fit_tree(X2, y2)

        if self.feature_names is not None:
            best_split_column_name = self.feature_names[best_split_column]
        else:
            best_split_column_name = None

        return Node(
            left_tree,
            right_tree,
            best_split_column_name,
            best_split_column,
            best_split_value,
        )

    def _check_purity(self, y: Union[np.ndarray, pd.Series]) -> bool:
        """
        Check if the target variable is pure (contains only one class).

        Parameters
        ----------
        y : Union[np.ndarray, pd.Series]
            The target variable.

        Returns
        -------
        bool
            True if the target variable is pure, False otherwise.
        """
        return len(np.unique(y)) == 1

    def _classify_data(self, y: Union[np.ndarray, pd.Series]) -> Any:
        """
        Return the classification of a leaf based on the most frequent class.

        Parameters
        ----------
        y : Union[np.ndarray, pd.Series]
            The target variable.

        Returns
        -------
        Any
            The most frequent class in the target variable.
        """
        classes, count = np.unique(y, return_counts=True)
        index = count.argmax()
        return classes[index]

    def _overall_entropy(
        self,
        y1: Union[np.ndarray, pd.Series],
        y2: Union[np.ndarray, pd.Series],
    ) -> float:
        """
        Calculate the overall entropy of two sets.

        Parameters
        ----------
        y1 : Union[np.ndarray, pd.Series]
            The target variable of the first set.
        y2 : Union[np.ndarray, pd.Series]
            The target variable of the second set.

        Returns
        -------
        float
            The calculated overall entropy.
        """
        len_1, len_2 = len(y1), len(y2)

        entropy_1 = self._calculate_entropy(y1)
        entropy_2 = self._calculate_entropy(y2)

        return (len_1 * entropy_1 + len_2 * entropy_2) / (len_1 + len_2)

    def _calculate_entropy(self, y: Union[np.ndarray, pd.Series]) -> float:
        """
        Calculate the entropy of a target variable.

        Parameters
        ----------
        y : Union[np.ndarray, pd.Series]
            The target variable.

        Returns
        -------
        float
            The calculated entropy.
        """
        _, counts = np.unique(y, return_counts=True)

        probs = counts / counts.sum()

        return -np.sum(probs * np.log2(probs))

    def _best_split(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        potential_splits: Dict[int, List[Any]],
    ) -> Tuple[int, float]:
        """
        Find the best split column and value for the data.

        Parameters
        ----------
        X : Union[np.ndarray, pd.DataFrame]
            The feature matrix.
        y : Union[np.ndarray, pd.Series]
            The target variable.
        potential_splits : Dict[int, List[Any]]
            The potential split values for each column.

        Returns
        -------
        Tuple[int, float]
            The index of the best split column and the corresponding split value.
        """
        X = np.array(X)
        y = np.array(y)

        min_entropy = None

        for index in potential_splits:
            for value in potential_splits[index]:
                X1, y1, X2, y2 = self._split_data(X, y, index, value)
                entropy = self._overall_entropy(y1, y2)

                if min_entropy is None:
                    min_entropy = entropy

                if entropy <= min_entropy:
                    min_entropy = entropy
                    best_split_column = index
                    best_split_value = value

        return best_split_column, best_split_value


class DecisionTreeRegressor(BaseTree):
    """
    Decision Tree Regressor Algorithm
    """

    def __init__(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        min_samples_leaf: Union[int, None] = 5,
        max_depth: Union[int, None] = 10,
    ) -> None:
        """
        Initialize a DecisionTreeRegressor object.

        Parameters
        ----------
        X : Union[np.ndarray, pd.DataFrame]
            The feature matrix.
        y : Union[np.ndarray, pd.Series]
            The target variable.
        """
        self.X = X
        self.y = y
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth

        try:
            self.feature_names = self.X.columns
        except:
            self.feature_names = None

    def fit(self) -> None:
        """
        Fit the decision tree on the provided data.
        """
        self._fitted_tree = self._fit_tree(self.X, self.y)

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Predict the class labels for the input data.

        Parameters
        ----------
        X : Union[np.ndarray, pd.DataFrame]
            The feature matrix for prediction.

        Returns
        -------
        np.ndarray
            The predicted class labels.
        """
        X = np.array(X)
        predictions = []

        for line in X:
            predictions.append(self._predict_line(line))

        return np.array(predictions)

    def _predict_line(
        self, X: Union[np.ndarray, pd.DataFrame], tree: Union[Node, None] = None
    ) -> Any:
        """
        Predict the class label for a single input instance.

        Parameters
        ----------
        X : Union[np.ndarray, pd.DataFrame]
            The feature matrix for prediction.
        tree : Union[Node, None], optional
            The decision tree node to start prediction from, by default None.

        Returns
        -------
        Any
            The predicted class label.
        """
        if tree is None:
            tree = self._fitted_tree

        if not isinstance(tree, Node):
            return tree

        if X[tree.split_metric_index].item() <= tree.split_value:
            return self._predict_line(X, tree.left_node)
        else:
            return self._predict_line(X, tree.right_node)

    def _mse(self, y_pred: Union[np.ndarray, pd.Series]) -> float:
        y_mean = np.mean(y_pred)
        return np.mean((y_pred - y_mean) ** 2)

    def _overall_mse(
        self,
        y1: Union[np.ndarray, pd.Series],
        y2: Union[np.ndarray, pd.Series],
    ) -> float:
        """
        Calculate the overall mse of two sets.

        Parameters
        ----------
        y1 : Union[np.ndarray, pd.Series]
            The target variable of the first set.
        y2 : Union[np.ndarray, pd.Series]
            The target variable of the second set.

        Returns
        -------
        float
            The calculated overall mse.
        """
        len_1, len_2 = len(y1), len(y2)

        mse_1 = self._mse(y1)
        mse_2 = self._mse(y2)

        return (len_1 * mse_1 + len_2 * mse_2) / (len_1 + len_2)

    def _best_split(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        potential_splits: Dict[int, List[Any]],
    ) -> Tuple[int, float]:
        """
        Find the best split column and value for the data.

        Parameters
        ----------
        X : Union[np.ndarray, pd.DataFrame]
            The feature matrix.
        y : Union[np.ndarray, pd.Series]
            The target variable.
        potential_splits : Dict[int, List[Any]]
            The potential split values for each column.

        Returns
        -------
        Tuple[int, float]
            The index of the best split column and the corresponding split value.
        """
        X = np.array(X)
        y = np.array(y)

        min_mse = None

        for index in potential_splits:
            for value in potential_splits[index]:
                X1, y1, X2, y2 = self._split_data(X, y, index, value)
                mse = self._overall_mse(y1, y2)

                if min_mse is None:
                    min_mse = mse

                if mse <= min_mse:
                    min_mse = mse
                    best_split_column = index
                    best_split_value = value

        return best_split_column, best_split_value

    def _estimate(self, y: Union[np.ndarray, pd.Series]) -> float:
        return float(np.mean(y))

    def _fit_tree(
        self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]
    ) -> Union[Node, float]:
        """
        Recursively build the decision tree.

        Parameters
        ----------
        X : Union[np.ndarray, pd.DataFrame]
            The feature matrix.
        y : Union[np.ndarray, pd.Series]
            The target variable.

        Returns
        -------
        Union[Node, float]
            The root node of the built decision tree or predicted value for a leaf.
        """
        depth = 0

        if self.max_depth is None:
            self.max_depth = np.inf

        if self.min_samples_leaf is None:
            self.min_samples_leaf = 0

        if len(X) <= self.min_samples_leaf or depth >= self.max_depth:
            predicted_value = self._estimate(y)
            return predicted_value
        else:
            depth += 1
            potential_splits = self._get_potential_splits(X)
            best_split_column, best_split_value = self._best_split(
                X, y, potential_splits
            )
            X1, y1, X2, y2 = self._split_data(X, y, best_split_column, best_split_value)

        left_tree = self._fit_tree(X1, y1)
        right_tree = self._fit_tree(X2, y2)

        if self.feature_names is not None:
            best_split_column_name = self.feature_names[best_split_column]
        else:
            best_split_column_name = None

        return Node(
            left_tree,
            right_tree,
            best_split_column_name,
            best_split_column,
            best_split_value,
        )
