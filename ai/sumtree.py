# ai/sumtree.py
import numpy as np

class SumTree:
    """
    SumTree 是一個二叉樹結構，用於高效地儲存優先級並進行採樣。
    """
    def __init__(self, capacity):
        self.capacity = capacity  # 緩衝區容量
        self.tree = np.zeros(2 * capacity - 1) # 樹的陣列，儲存優先級和總和
        self.data = np.array([None] * capacity) # 儲存實際經驗的陣列
        self.data_pointer = 0 # 指向下一個可用於儲存經驗的位置

    def add(self, priority, data):
        """
        添加新數據和其優先級到 SumTree 中。
        """
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data
        self.update(tree_idx, priority) # 更新葉節點和其父節點的總和

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0 # 如果達到容量，從頭開始覆蓋

    def update(self, tree_idx, priority):
        """
        更新樹中某個節點的優先級。
        """
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        while tree_idx != 0: # 向上傳播更新，直到根節點
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        """
        從根節點向下查找，根據隨機值 v 選擇一個葉節點。
        """
        parent_idx = 0
        while True:
            left_child_idx = 2 * parent_idx + 1
            right_child_idx = left_child_idx + 1
            if left_child_idx >= len(self.tree):
                leaf_idx = parent_idx
                break
            else:
                if v <= self.tree[left_child_idx]:
                    parent_idx = left_child_idx
                else:
                    v -= self.tree[left_child_idx]
                    parent_idx = right_child_idx
        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_priority(self):
        """
        返回樹的總優先級（根節點的值）。
        """
        return self.tree[0]