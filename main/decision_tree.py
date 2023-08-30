from typing import List, Union, Any, Dict, Tuple
import pandas as pd
import numpy as np

class Node:
    def __init__(
        self, left_node, right_node, split_metric: int, split_value: float
    ) -> None:
        """
        Initialize a Node object in a decision tree.

        Parameters
        ----------
        left_node : Node
            The left child node.
        right_node : Node
            The right child node.
        split_metric : int
            The index of the feature used for splitting.
        split_value : float
            The value of the feature used for splitting.
        """
        self.left_node = left_node
        self.right_node = right_node
        self.split_metric = split_metric
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

class DecisionTreeClassifier:
    def __init__(
        self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]
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

    def fit(self):
        """
        Fit the decision tree on the provided data.
        """
        self._fitted_tree = self._fit_tree(self.X, self.y)

    def _fit_tree(
        self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]
    ):
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
        Node
            The root node of the built decision tree.
        """
        if self.check_purity(y):
            classification = self.classify_data(y)
            return classification
        else:
            potential_splits = self.get_potential_splits(X)
            best_split_column, best_split_value = self.best_split(
                X, y, potential_splits
            )
            X1, y1, X2, y2 = self.split_data(X, y, best_split_column, best_split_value)

        left_tree = self._fit_tree(X1, y1)
        right_tree = self._fit_tree(X2, y2)
        return Node(left_tree, right_tree, best_split_column, best_split_value)

    def check_purity(self, y: Union[np.ndarray, pd.Series]) -> bool:
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

    def classify_data(self, y: Union[np.ndarray, pd.Series]) -> Any:
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

    def get_potential_splits(
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

    def split_data(
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

    def overall_entropy(
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

        entropy_1 = self.calculate_entropy(y1)
        entropy_2 = self.calculate_entropy(y2)

        return (len_1 * entropy_1 + len_2 * entropy_2) / (len_1 + len_2)

    def calculate_entropy(self, y: Union[np.ndarray, pd.Series]) -> float:
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

    def best_split(
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
                X1, y1, X2, y2 = self.split_data(X, y, index, value)
                entropy = self.overall_entropy(y1, y2)

                if min_entropy == None:
                    min_entropy = entropy

                if entropy <= min_entropy:
                    min_entropy = entropy
                    best_split_column = index
                    best_split_value = value

        return best_split_column, best_split_value
