class TreeNode(object):
    def __init__(self,data):
        self.data = data
        self.left = None
        self.right = None

class BinaryTree(object):
    self.root = None
    def __init__(self,arr,index):

        root = createBinartTree(arr, index)

    def createBinartTree(arr,index):
        treeNode = None
        if index< arr.length:
            treeNode = TreeNode(arr , index)
            treeNode.left = createBinartTree(arr,2*index+1)
            treeNode.right = createBinartTree(arr,2*index+2)
        
        return treeNode

    if __name__ = "__main__":
        arr = [1,2,3,4,5,6,7,8,9]
        index =0
        binaryTree = BinaryTree(arr,index)
        print(binaryTree.root)
        print(arr)