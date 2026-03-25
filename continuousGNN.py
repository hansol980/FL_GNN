import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch_geometric.datasets import WikiCS
from torch_geometric.nn import GCNConv

# WikiCS 데이터셋 로드: 각 노드의 특성(Feature)이 GloVe 기반의 300차원 연속형 밀집(Dense Continuous) 벡터입니다.
# Cora의 sparse(0과 1)한 BoW 한계를 극복하고, 현실적인 유출 공격 가능성을 테스트하기에 안성맞춤입니다.
print("데이터셋 다운로드 및 로드 중 (WikiCS)...")
dataset = WikiCS(root='/tmp/WikiCS')
data = dataset[0]

# WikiCS는 2D 형태의 분할 마스크(여러 개의 반복 실험용)를 제공하므로 첫 번째 분할을 선택합니다.
train_mask = data.train_mask[:, 0]
test_mask = data.test_mask

class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        # input : num_node_features(300), hidden : 32, output : num_classes(10)
        self.conv1 = GCNConv(dataset.num_node_features, 32)
        self.conv2 = GCNConv(32, dataset.num_classes)

    def forward_with_x(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.tanh(x)
        x = F.dropout(x, p=0.5, training=self.training)
        
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
        
    def forward(self, data):
        return self.forward_with_x(data.x, data.edge_index)

def collect_original_gradients(model, x, edge_index, y, target_idx):
    model.eval()
    model.zero_grad()
    out = model.forward_with_x(x, edge_index)
    target_loss = F.nll_loss(out[target_idx:target_idx + 1], y[target_idx:target_idx + 1])
    grads = torch.autograd.grad(target_loss, model.parameters(), create_graph=False)
    return [g.detach().clone() for g in grads]

def infer_label_from_gradients(model, true_grads, num_classes):
    target_bias_grad = None
    for (name, _), grad in zip(model.named_parameters(), true_grads):
        if name.endswith('conv2.bias') and grad.dim() == 1 and grad.numel() == num_classes:
            target_bias_grad = grad
            break

    if target_bias_grad is None:
        for grad in true_grads:
            if grad.dim() == 1 and grad.numel() == num_classes:
                target_bias_grad = grad
                break

    if target_bias_grad is None:
        raise RuntimeError('iDLG label inference failed: classifier bias gradient not found.')

    return int(torch.argmin(target_bias_grad).item())

def run_idlg_attack_continuous(model, data, target_idx, iters=500, attack_lr=0.01):
    model.eval()
    for p in model.parameters():
        p.requires_grad_(True)

    true_grads = collect_original_gradients(model, data.x, data.edge_index, data.y, target_idx)
    inferred_label = infer_label_from_gradients(model, true_grads, dataset.num_classes)

    fixed_x = data.x.detach().clone()
    target_mask = torch.zeros((data.num_nodes, 1), device=data.x.device)
    target_mask[target_idx] = 1.0

    # GloVe 특징은 대략 0을 중심으로 분포하는 밀집형 연속 벡터(Dense Continuous Vector)이므로 정규 분포로 초기화합니다.
    # 이전 논리에서의 0과 1 사이의 제약(sigmoid)은 피쳐 특성 상 제거하고 대신 L2 정규화를 줍니다.
    dummy_x = torch.randn((1, data.num_features), device=data.x.device, requires_grad=True)
    dummy_optimizer = torch.optim.Adam([dummy_x], lr=attack_lr)

    target_label_tensor = torch.tensor([inferred_label], dtype=torch.long, device=data.x.device)

    for it in range(iters):
        dummy_optimizer.zero_grad()

        expanded_dummy_x = dummy_x.expand(data.num_nodes, -1)
        mixed_x = fixed_x * (1.0 - target_mask) + expanded_dummy_x * target_mask

        out = model.forward_with_x(mixed_x, data.edge_index)
        dummy_loss = F.nll_loss(out[target_idx:target_idx + 1], target_label_tensor)

        dummy_grads = torch.autograd.grad(dummy_loss, model.parameters(), create_graph=True)
        # Gradient 매칭 (유클리드 거리 편차 제곱)
        grad_diff = sum(((dg - tg) ** 2).sum() for dg, tg in zip(dummy_grads, true_grads))

        # 밀집 특성(Dense Feature)에 적합한 L2 Penalty 적용
        # L1 Norm은 희소성(Sparsity)을 주지만, 연속형 특징에는 벡터 크기가 터지지 않게 잡아주는 L2 가 적절합니다.
        alpha_l2 = 0.005 
        l2_loss = alpha_l2 * torch.norm(dummy_x, p=2)

        total_loss = grad_diff + l2_loss
        total_loss.backward()
        dummy_optimizer.step()

        if it % 100 == 0 or it == iters - 1:
            print(f"iDLG iter {it}, grad diff: {grad_diff.item():.6f}, l2 loss: {l2_loss.item():.6f}")

    true_label = data.y[target_idx].item()
    feature_mse = F.mse_loss(dummy_x.detach(), data.x[target_idx:target_idx + 1]).item()
    cosine_sim = F.cosine_similarity(dummy_x.detach(), data.x[target_idx:target_idx + 1]).item()

    return {
        'target_idx': int(target_idx),
        'true_label': int(true_label),
        'inferred_label': int(inferred_label),
        'feature_mse': float(feature_mse),
        'feature_cosine_similarity': float(cosine_sim),
        'recovered_feature': dummy_x.detach().cpu(),
        'true_feature': data.x[target_idx:target_idx + 1].detach().cpu()
    }


def visualize_reconstruction(true_feature, recovered_feature, save_path='wikics_reconstruction.png', top_k=50):
    true_vec = true_feature.view(-1)
    rec_vec = recovered_feature.view(-1)

    top_idx = torch.topk(torch.abs(true_vec), k=top_k).indices
    x_axis = torch.arange(top_k).numpy()
    true_top = true_vec[top_idx].numpy()
    rec_top = rec_vec[top_idx].numpy()

    true_np = true_vec.numpy()
    rec_np = rec_vec.numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(x_axis, true_top, label='True Feature (WikiCS)', marker='o', linewidth=1.2)
    axes[0].plot(x_axis, rec_top, label='Recovered Feature', marker='x', linewidth=1.2)
    axes[0].set_title(f'Top-{top_k} Features | True vs Recovered')
    axes[0].set_xlabel('Feature Rank (by magnitude)')
    axes[0].set_ylabel('Feature Value')
    axes[0].legend()

    axes[1].scatter(true_np, rec_np, alpha=0.5, s=15, color='orange')
    min_v = min(true_np.min(), rec_np.min())
    max_v = max(true_np.max(), rec_np.max())
    axes[1].plot([min_v, max_v], [min_v, max_v], linestyle='--', linewidth=1.2, color='blue')
    axes[1].set_title('All Dimensions Correlation')
    axes[1].set_xlabel('True Feature Value')
    axes[1].set_ylabel('Recovered Feature Value')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
model = GCN().to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# 첫번째 train_mask 기준 가장 앞쪽의 노드를 타겟으로 선정
target_idx = int(train_mask.nonzero(as_tuple=True)[0][0].item())

# ================= 모델 훈련 =================
print("\n--- GCN 모델 훈련 시작 (WikiCS) ---")
model.train()
for epoch in range(500): # 빠른 확인을 위해 100회만
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[train_mask], data.y[train_mask])
    loss.backward()
    optimizer.step()
    
    if epoch % 50 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

model.eval()
pred = model(data).argmax(dim=1)
correct = (pred[test_mask] == data.y[test_mask]).sum()
acc = int(correct) / int(test_mask.sum())
print(f'\n최종 테스트 정확도: {acc:.4f}')

# ================= 훈련 후 상태 공격 =================
print("\n[훈련 완료(Trained) 신경망 기밀성 테스트 시작 - WikiCS]")
idlg_result = run_idlg_attack_continuous(model, data, target_idx=target_idx, iters=500, attack_lr=0.05)

print("\n[훈련 완료 신경망 iDLG 결과]")
print(f"실제 레이블: {idlg_result['true_label']}")
print(f"추정 레이블: {idlg_result['inferred_label']}")
print(f"특징 복원 MSE: {idlg_result['feature_mse']:.6f}")
print(f"특징 코사인 유사도: {idlg_result['feature_cosine_similarity']:.6f}")

visualize_reconstruction(
    true_feature=idlg_result['true_feature'],
    recovered_feature=idlg_result['recovered_feature'],
    save_path='idlg_wikics_reconstruction_trained.png',
    top_k=50,
)
print("훈련 완료 시스템 공격 결과 시각화 저장 완료: idlg_wikics_reconstruction_trained.png")
