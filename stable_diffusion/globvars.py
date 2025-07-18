# Training
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
val = {
}