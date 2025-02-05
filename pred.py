import torch
from enformer_pytorch import from_pretrained
from scipy.stats import pearsonr

# Load the fine-tuned model
#model = from_pretrained('EleutherAI/enformer-official-rough')#.cuda()
#model.load_state_dict(torch.load("fine_tuned_enformer.pth"))
model = torch.load("fine_tuned_enformer.pth")

# Set model to evaluation mode
model.eval()
print("Fine-tuned model loaded successfully.")

# Load the sequence and target
seq = torch.load("fine_tuned_seq.pt")#.cuda()
print(seq[:10])
target_tensor = torch.load("fine_tuned_targets.pt")#.cuda()

print(" data loaded successfully.")


# Get predictions from the model
# model.eval()
# with torch.no_grad():
#     predictions = model(seq)#['human']#.cpu().numpy()
#     targets = target_tensor#.cpu().numpy()

# # Compute Pearson correlation coefficient
# corr, _ = pearsonr(predictions.flatten(), targets.flatten())

# print(f"Pearson Correlation: {corr:.4f}")

########
enformer = from_pretrained('EleutherAI/enformer-official-rough', use_tf_gamma = False)#.cuda()
enformer.eval()

#data = torch.load('./data/test-sample.pt', map_location=torch.device('cpu'))
#seq, target = data['sequence'], data['target']

with torch.no_grad():
    corr_coef = enformer(
        seq,
        target = target_tensor,
        return_corr_coef = True,
        head = 'human'
    )

print(corr_coef)
assert corr_coef > 0.1