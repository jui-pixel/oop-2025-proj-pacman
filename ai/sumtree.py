# ai/sumtree.py
import numpy as np

class SumTree:
    """
    SumTree 是一個用於優先級經驗回放（Prioritized Experience Replay, PER）的二叉樹結構，
    用於高效儲存和採樣優先級數據。

    原理：
    - SumTree 是一個完全二叉樹，用於在深度強化學習中實現優先級經驗回放。
    - 每個葉節點儲存一個經驗的優先級（通常為 TD 誤差），父節點儲存其子節點優先級之和。
    - 根節點的優先級總和等於所有經驗的優先級總和，方便按優先級比例採樣。
    - 採樣效率：通過隨機值 v 在樹中進行二分查找，時間複雜度為 O(log N)，N 為葉節點數。
    - 數據結構：
      - self.tree：陣列表示的二叉樹，長度為 2*capacity - 1，儲存優先級和總和。
      - self.data：儲存實際經驗數據（例如轉換元組），長度為 capacity。
      - self.data_pointer：指向下一個可用儲存位置，實現環形緩衝區。
    - 優先級採樣公式：p(i) = priority_i / total_priority，採樣概率與優先級成正比。
    """
    def __init__(self, capacity):
        """
        初始化 SumTree 結構。

        Args:
            capacity (int): 緩衝區的最大容量，即葉節點數量。

        原理：
        - 完全二叉樹的節點數計算：
          - 葉節點數：capacity
          - 總節點數：2 * capacity - 1（包括所有父節點和葉節點）
        - self.tree 儲存優先級，葉節點索引從 capacity - 1 開始。
        - self.data 儲存實際經驗數據，與葉節點一一對應。
        """
        self.capacity = capacity  # 緩衝區容量（葉節點數）
        self.tree = np.zeros(2 * capacity - 1)  # 樹陣列，儲存優先級和父節點總和
        self.data = np.array([None] * capacity)  # 數據陣列，儲存經驗數據
        self.data_pointer = 0  # 指向下一個可用儲存位置

    def add(self, priority, data):
        """
        添加新經驗數據及其優先級到 SumTree 中。

        原理：
        - 新經驗存入 self.data，並將對應優先級存入葉節點。
        - 更新葉節點的優先級後，向上傳播更新父節點的優先級總和。
        - 使用環形緩衝區策略，當 data_pointer 達到容量時從頭覆蓋。
        - 葉節點索引計算公式：tree_idx = data_pointer + capacity - 1

        Args:
            priority (float): 經驗的優先級，通常為 TD 誤差加小常數（例如 |TD_error| + ε）。
            data: 經驗數據（例如轉換元組 Transition）。
        """
        tree_idx = self.data_pointer + self.capacity - 1  # 計算葉節點索引
        self.data[self.data_pointer] = data  # 儲存經驗數據
        self.update(tree_idx, priority)  # 更新優先級

        self.data_pointer += 1  # 更新指針
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0  # 達到容量時從頭開始覆蓋

    def update(self, tree_idx, priority):
        """
        更新指定葉節點的優先級，並傳播到根節點。

        原理：
        - 更新葉節點的優先級後，計算優先級變化量：change = new_priority - old_priority
        - 從葉節點向上更新父節點的優先級總和，直到根節點。
        - 父節點索引計算公式：parent_idx = (tree_idx - 1) // 2
        - 這種更新方式確保樹的總和始終反映所有葉節點的優先級之和。

        Args:
            tree_idx (int): 葉節點的索引。
            priority (float): 新優先級值。
        """
        change = priority - self.tree[tree_idx]  # 計算優先級變化量
        self.tree[tree_idx] = priority  # 更新葉節點優先級
        while tree_idx != 0:  # 向上傳播更新
            tree_idx = (tree_idx - 1) // 2  # 計算父節點索引
            self.tree[tree_idx] += change  # 更新父節點總和

    def get_leaf(self, v):
        """
        根據隨機值 v 從樹中查找對應的葉節點。

        原理：
        - 採樣過程模擬按優先級比例隨機選擇經驗。
        - 從根節點開始，根據隨機值 v（範圍 [0, total_priority]）進行二分查找：
          - 若 v ≤ 左子節點總和，進入左子樹。
          - 否則，v -= 左子節點總和，進入右子樹。
        - 最終找到葉節點，返回其索引、優先級和對應的經驗數據。
        - 數據索引計算公式：data_idx = leaf_idx - capacity + 1

        Args:
            v (float): 隨機值，範圍 [0, total_priority]。

        Returns:
            tuple: (leaf_idx, priority, data)
            - leaf_idx: 葉節點索引。
            - priority: 該葉節點的優先級。
            - data: 對應的經驗數據。
        """
        parent_idx = 0  # 從根節點開始
        while True:
            left_child_idx = 2 * parent_idx + 1  # 左子節點索引
            right_child_idx = left_child_idx + 1  # 右子節點索引
            if left_child_idx >= len(self.tree):  # 達到葉節點
                leaf_idx = parent_idx
                break
            else:
                if v <= self.tree[left_child_idx]:  # 進入左子樹
                    parent_idx = left_child_idx
                else:  # 進入右子樹
                    v -= self.tree[left_child_idx]
                    parent_idx = right_child_idx
        data_idx = leaf_idx - self.capacity + 1  # 計算數據索引
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_priority(self):
        """
        返回樹的總優先級（根節點的值）。

        原理：
        - 根節點（self.tree[0]）儲存所有葉節點優先級之和。
        - 用於計算採樣概率：p(i) = priority_i / total_priority。

        Returns:
            float: 所有經驗的優先級總和。
        """
        return self.tree[0]