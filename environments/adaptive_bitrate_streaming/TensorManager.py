import torch

class TensorManager:
    def __init__(self, max_rows=30, feature_dim=7):
        self.max_rows = max_rows
        self.feature_dim = feature_dim
        self.data_tensor = torch.zeros((self.max_rows, self.feature_dim))

    def add_data(self, new_data):
        """
        添加新数据到 tensor 中，保持最大行数为 30。
        
        :param new_data: 形状为 (1, 7) 的 tensor 或者列表
        """
        new_data_tensor = torch.tensor(new_data).unsqueeze(0)  # 确保是 (1, 7) 的形状
        if self.data_tensor.shape[0] >= self.max_rows:
            # 删除第一行，向上移动
            self.data_tensor = torch.cat((self.data_tensor[1:], new_data_tensor), dim=0)
        else:
            # 追加新数据
            self.data_tensor = torch.cat((self.data_tensor, new_data_tensor), dim=0)

    def get_tensor(self):
        """返回当前 tensor,如果没有满 30 行则抛出错误。"""
        if self.data_tensor.shape[0] < self.max_rows:
            raise ValueError(f"当前 tensor 只有 {self.data_tensor.shape[0]} 行，未满 {self.max_rows} 行。")
        return self.data_tensor