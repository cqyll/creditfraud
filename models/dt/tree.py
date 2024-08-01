class TreeNode:
    def __init__(self, feature_index = None, threshold = None, left = None, right = None, value = None, purity = None):
        """
        Initializes a tree node.

        Args:
            feature_index (int, optional): index of feature used for splitting. Defaults to None.
            threshold (float, optional): threshold value for split. Defaults to None.
            left (TreeNode, optional): Left child node. Defaults to None.
            right (TreeNode, optional): Right child node. Defaults to None.
            value (float, optional): Predicted value (if leaf node). Defaults to None.
            purity (float, optional): Purity measure of the node. Defaults to None.

        """
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.purity = purity