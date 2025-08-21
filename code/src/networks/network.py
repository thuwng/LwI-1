import torch
from torch import nn
from copy import deepcopy

class LLL_Net(nn.Module):
    def __init__(self, model, remove_existing_head=False):
        head_var = model.head_var
        assert type(head_var) == str
        assert not remove_existing_head or hasattr(model, head_var), \
            "Given model does not have a variable called {}".format(head_var)
        assert not remove_existing_head or type(getattr(model, head_var)) in [nn.Sequential, nn.Linear], \
            "Given model's head {} is not an instance of nn.Sequential or nn.Linear".format(head_var)
        super(LLL_Net, self).__init__()

        self.model = model
        last_layer = getattr(self.model, head_var)
        print("last_layer", last_layer)
        print(f"remove_existing_head: {remove_existing_head}")
        if remove_existing_head:
            if type(last_layer) == nn.Sequential:
                self.out_size = last_layer[-1].in_features
                del last_layer[-1]
            elif type(last_layer) == nn.Linear:
                self.out_size = last_layer.in_features
                setattr(self.model, head_var, nn.Sequential())
            print(f"Head removed, out_size: {self.out_size}")
        else:
            self.out_size = last_layer.out_features
            print(f"Keeping existing head, out_size: {self.out_size}")
        
        self.heads = nn.ModuleList()
        self.task_cls = torch.tensor([], dtype=torch.int64)
        self.task_offset = torch.tensor([0], dtype=torch.int64)
        self._initialize_weights()

    def add_head(self, num_outputs):
        print(f"Adding head with {num_outputs} classes")
        # Xóa tất cả head cũ để tránh nhầm lẫn
        self.heads = nn.ModuleList([nn.Linear(self.out_size, num_outputs, bias=False)])
        self.task_cls = torch.tensor([num_outputs], dtype=torch.int64)
        print("self.task_cls", self.task_cls)
        self.task_offset = torch.tensor([0], dtype=torch.int64)
        print("self.task_offset", self.task_offset)

    def forward(self, x, return_features=False, task_id=None):
        features = self.model(x, return_features=True)   # backbone trả về feature
        assert len(self.heads) > 0, "Cannot access any head"
        if task_id is not None:
            # Chỉ sử dụng head của task hiện tại
            y = self.heads[task_id](x)
            print(f"Head {task_id} output shape: {y.shape}")
            if return_features:
                return [y], x
            return [y]
        else:
            # Trả về đầu ra của tất cả head (cho đánh giá)
            y = [head(x) for head in self.heads]
            for i, out in enumerate(y):
                print(f"Head {i} output shape: {out.shape}")
            if return_features:
                return y, x
            return y

    def get_copy(self):
        return deepcopy(self.state_dict())

    def set_state_dict(self, state_dict):
        self.load_state_dict(deepcopy(state_dict))
        return

    def freeze_all(self):
        for param in self.parameters():
            param.requires_grad = False

    def freeze_backbone(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def freeze_bn(self):
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def _initialize_weights(self):
        pass