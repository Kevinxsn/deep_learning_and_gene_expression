import torch
from enformer_pytorch import from_pretrained

enformer = from_pretrained('EleutherAI/enformer-official-rough', use_tf_gamma = False)#.cuda()
enformer.eval()

data = torch.load('./data/test-sample.pt', map_location=torch.device('cpu'))
#print(data[:10])
seq, target = data['sequence'], data['target']
print(seq[:10])
print(target[:10])
print(seq.shape)
print(target.shape)

with torch.no_grad():
    corr_coef = enformer(
        seq,
        target = target,
        return_corr_coef = True,
        head = 'human'
    )
    predictions = enformer(seq)

print([predictions['human'].shape])
print(corr_coef)
assert corr_coef > 0.1
