import torch

class SelectiveScanObserver:
    def __init__(self):
        self.inputs = []
        self.outputs = []

    def hook_in(self, module, inputs):
        with torch.no_grad():
            x = inputs[0]
            self.inputs.append(x.detach().float().cpu())

    def hook_out(self, module, inputs, output):
        with torch.no_grad():
            self.outputs.append(output.detach().float().cpu())

    def clear(self):
        self.inputs.clear()
        self.outputs.clear()
