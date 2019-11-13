class TreeNode:
    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None


class BinaryTree:
    def __init__(self, arr, index):

        self.root = self.createBinartTree(arr, index)

    def createBinartTree(self, arr, index):
        treeNode = None
        if index < len(arr):
            treeNode = TreeNode(arr[index])
            treeNode.left = self.createBinartTree(arr, 2 * index + 1)
            treeNode.right = self.createBinartTree(arr, 2 * index + 2)

        return treeNode



if __name__ == "__main__":
    arr = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    index = 0
    binaryTree = BinaryTree(arr, index)
    print(binaryTree.root.data)
    print(arr)
