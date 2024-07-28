# Định nghĩa một lớp đại diện cho nút trong cây quyết định
class DecisionTreeNode:
    def __init__(self, is_leaf=False, decision=None, condition=None):
        self.is_leaf = is_leaf
        self.decision = decision
        self.condition = condition
        self.left = None
        self.right = None

class DecisionTree:
    def __init__(self, root):
        self.root = root

# Tạo nút theo thị trường
market_node = DecisionTreeNode()

# Điều kiện đầu tiên là thị trường tốt
market_node.condition = "Thị trường"

# Nếu thị trường xấu, không học
market_node.left = DecisionTreeNode(is_leaf=True, decision="Không học")

# Điều kiện tiếp theo dựa trên niềm đam mê
passion_node = DecisionTreeNode()
passion_node.condition = "Đam mê"

# Nếu niềm đam mê thấp, không học
passion_node.left = DecisionTreeNode(is_leaf=True, decision="Không học")

# Nếu niềm đam mê cao, học
passion_node.right = DecisionTreeNode(is_leaf=True, decision="Học")

market_node.right = passion_node

decision_tree = DecisionTree(market_node)

def predict(tree, market, passion):
    if tree.is_leaf:
        return tree.decision
    if tree.condition == "Thị trường":
        if market == "Tốt":
            return predict(tree.right, market, passion)
        else:
            return predict(tree.left, market, passion)
    elif tree.condition == "Đam mê":
        if passion == "Có":
            return predict(tree.right, market, passion)
        else:
            return predict(tree.left, market, passion)

# Sử dụng cây quyết định để dự đoán
market = "Tốt"  # Thị trường tốt
passion = "Có"  # Niềm đam mê cao

decision = predict(decision_tree.root, market, passion)
print(f"Quyết định là: {decision}")
