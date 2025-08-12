# Training
train = {}
"""
to capture grads, set breakpoint at /usr/local/lib/python3.10/dist-packages/torch/cuda/amp/grad_scaler.py line 1742
when above breakpoint is reached, in debugger go to /usr/local/lib/python3.10/dist-packages/lightning/pytorch/core/module.py line 1742 context (optimizer_step)
then run:
import globvars
from safetensors.torch import save_file
g = {k:v.grad for k,v in self.model.diffusion_model.named_parameters()}
globvars.train["out.2.weight"] = g["out.2.weight"]
globvars.train["out.2.bias"] = g["out.2.bias"]
save_file(globvars.train, "checkpoints/train0.safetensors")
"""

unet_inputs = {}
export_tensors = {}
sample_attn = False
train_steps = {
  "batch_npy": [],
  "t": [],
  "noise": [],
  "latent_randn": [],
  "loss": [],
}
prompts = []
step = 0

# Validation
step_num=0
val = {
}

# mixed precision
mixed = {}

capture_layernorm=True
capture_groupnorm=True
capture_softmax=True