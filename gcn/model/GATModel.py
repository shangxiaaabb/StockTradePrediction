'''
Author: h-jie huangjie20011001@163.com
Date: 2024-06-23 16:19:53
'''

import torch.nn as nn
import torch.nn.functional as F
import torch

class GraphAttentionLayer(nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
