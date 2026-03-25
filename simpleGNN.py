import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv

dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]

class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        # input : num_node_features(1433), hidden : 16, output : num_classes(7)
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward_with_x(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index # edge_index는 COO 형식의 인접 행렬을 나타냄
        return self.forward_with_x(x, edge_index)


def collect_original_gradients(model, x, edge_index, y, target_idx):
    model.eval()
    model.zero_grad()
    out = model.forward_with_x(x, edge_index)
    target_loss = F.nll_loss(out[target_idx:target_idx + 1], y[target_idx:target_idx + 1])
    grads = torch.autograd.grad(target_loss, model.parameters(), create_graph=False)
    return [g.detach().clone() for g in grads]


def infer_label_from_gradients(model, true_grads, num_classes):
    # iDLG: for single-sample CE/NLL loss, argmin of classifier bias grad gives the true label.
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


def run_idlg_attack(model, data, target_idx, iters=300, attack_lr=0.1):
    model.eval()
    for p in model.parameters():
        p.requires_grad_(True)

    true_grads = collect_original_gradients(model, data.x, data.edge_index, data.y, target_idx)
    inferred_label = infer_label_from_gradients(model, true_grads, dataset.num_classes)

    fixed_x = data.x.detach().clone()
    target_mask = torch.zeros((data.num_nodes, 1), device=data.x.device)
    target_mask[target_idx] = 1.0

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
        grad_diff = sum(((dg - tg) ** 2).sum() for dg, tg in zip(dummy_grads, true_grads))

        grad_diff.backward()
        dummy_optimizer.step()

        if it % 50 == 0 or it == iters - 1:
            print(f"iDLG iter {it}, grad diff: {grad_diff.item():.6f}")

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


def visualize_reconstruction(true_feature, recovered_feature, save_path='reconstruction_compare.png', top_k=30):
    true_vec = true_feature.view(-1)
    rec_vec = recovered_feature.view(-1)

    top_k = min(top_k, true_vec.numel())
    top_idx = torch.topk(torch.abs(true_vec), k=top_k).indices

    x_axis = torch.arange(top_k).numpy()
    true_top = true_vec[top_idx].numpy()
    rec_top = rec_vec[top_idx].numpy()

    true_np = true_vec.numpy()
    rec_np = rec_vec.numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(x_axis, true_top, label='True Feature', marker='o', linewidth=1.2)
    axes[0].plot(x_axis, rec_top, label='Recovered Feature', marker='x', linewidth=1.2)
    axes[0].set_title(f'Top-{top_k} | True vs Recovered')
    axes[0].set_xlabel('Feature Rank (by |true value|)')
    axes[0].set_ylabel('Feature Value')
    axes[0].legend()

    axes[1].scatter(true_np, rec_np, alpha=0.4, s=10)
    min_v = min(true_np.min(), rec_np.min())
    max_v = max(true_np.max(), rec_np.max())
    axes[1].plot([min_v, max_v], [min_v, max_v], linestyle='--', linewidth=1.0)
    axes[1].set_title('All Dimensions Correlation')
    axes[1].set_xlabel('True Feature')
    axes[1].set_ylabel('Recovered Feature')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
model = GCN().to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()
for epoch in range(500):
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    
    if epoch % 20 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

model.eval()
pred = model(data).argmax(dim=1)
correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
acc = int(correct) / int(data.test_mask.sum())
print(f'\n최종 테스트 정확도: {acc:.4f}')

target_idx = int(data.train_mask.nonzero(as_tuple=True)[0][0].item())

idlg_result = run_idlg_attack(model, data, target_idx=target_idx, iters=300, attack_lr=0.1)

print("\n[iDLG 결과]")
print(f"실제 레이블: {idlg_result['true_label']}")
print(f"추정 레이블(iDLG): {idlg_result['inferred_label']}")
print(f"특징 복원 MSE: {idlg_result['feature_mse']:.6f}")
print(f"특징 코사인 유사도: {idlg_result['feature_cosine_similarity']:.6f}")

visualize_reconstruction(
    true_feature=idlg_result['true_feature'],
    recovered_feature=idlg_result['recovered_feature'],
    save_path='idlg_reconstruction_compare.png',
    top_k=30,
)
print("시각화 저장 완료: idlg_reconstruction_compare.png")