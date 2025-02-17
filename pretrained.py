import torch
from enformer_pytorch import from_pretrained
from enformer_pytorch import seq_indices_to_one_hot
from scipy.stats import pearsonr

# Load the pretrained Enformer model
enformer = from_pretrained('EleutherAI/enformer-official-rough', use_tf_gamma=False)
enformer.eval()  # Set to evaluation mode
print("Pretrained model loaded successfully.")

# Load the sequence and target data
# One-hot encode: (batch_size, 4, seq_length) 
seq = torch.load("fine_tuned_seq.pt")
one_hot_seq = seq_indices_to_one_hot(seq)[0] # Enformer expects 4 classes
target_tensor = torch.load("fine_tuned_targets.pt")

print(one_hot_seq.shape)
print(target_tensor.shape)

# Run the pretrained model on the input sequence
# with torch.no_grad():
#     predictions = enformer(one_hot_seq)['human']  # Extract human predictions
with torch.no_grad():
    corr_coef = enformer(
        seq,
        target = target_tensor,
        return_corr_coef = True,
        head = 'human'
    )

print(corr_coef)

# print(f"Raw Model Output Shape: {predictions.shape}") # Check dimensions
# print(target_tensor.shape)

# import torch.nn.functional as F

# # Downsample predictions using adaptive pooling
# predictions = F.adaptive_avg_pool2d(predictions, (574, 670))

# # Convert predictions and targets to numpy arrays
# predictions_np = predictions.cpu().numpy().flatten()
# targets_np = target_tensor.cpu().numpy().flatten()

# predictions = -((predictions_np - predictions_np.mean()) / predictions_np.std())
# targets = (targets_np - targets_np.mean()) / targets_np.std()
# # Compute Pearson correlation
# corr_pretrained, _ = pearsonr(predictions, targets)
#print(f"Pearson Correlation (Pretrained Model): {corr_pretrained:.4f}")