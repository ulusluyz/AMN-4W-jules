import torch
import torch.nn as nn
import torch.nn.functional as F

class AMN4W(nn.Module):
    """
    Independent AMN-4W Architecture with dynamic relational memory.
    S(t+1) = gamma(t)S(t) + eta(t)[v(t) - S(t)k(t)]k(t)^T
    """
    def __init__(self, vocab_size, embed_dim, memory_size, max_seq_len):
        super(AMN4W, self).__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.memory_size = memory_size

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_seq_len * 2, embed_dim)

        self.fc_k = nn.Linear(embed_dim, memory_size)
        self.fc_v = nn.Linear(embed_dim, memory_size)
        self.fc_eta = nn.Linear(embed_dim, 1)
        self.fc_gamma = nn.Linear(embed_dim, 1)
        self.fc_q = nn.Linear(embed_dim, memory_size)

        self.fc_out1 = nn.Linear(embed_dim + memory_size, embed_dim)
        self.fc_out2 = nn.Linear(embed_dim, vocab_size)

    def forward(self, x, memory_state=None):
        b, seq_len = x.size()
        device = x.device

        if memory_state is None:
            memory_state = torch.zeros(b, self.memory_size, self.memory_size).to(device)

        pos = torch.arange(0, seq_len, dtype=torch.long).unsqueeze(0).expand(b, seq_len).to(device)

        emb = self.embedding(x) + self.pos_embedding(pos)

        out_logits = []

        for t in range(seq_len):
            xt = emb[:, t, :]

            k_t = self.fc_k(xt)
            v_t = self.fc_v(xt)

            # Normalizing keys and values for memory stability
            k_t = F.normalize(k_t, p=2, dim=-1)
            v_t = F.normalize(v_t, p=2, dim=-1)

            # Gating mechanisms bounded between 0 and 1
            eta_t = torch.sigmoid(self.fc_eta(xt))
            gamma_t = torch.sigmoid(self.fc_gamma(xt))

            # Reading for update step: S(t)k(t)
            S_k = torch.bmm(memory_state, k_t.unsqueeze(-1)).squeeze(-1)

            # Diff: v(t) - S(t)k(t)
            diff = v_t - S_k

            # Update term: [v(t) - S(t)k(t)]k(t)^T
            update_term = torch.bmm(diff.unsqueeze(-1), k_t.unsqueeze(1))

            # Memory state update
            memory_state = gamma_t.unsqueeze(-1) * memory_state + eta_t.unsqueeze(-1) * update_term

            # Read from memory for output
            q_t = self.fc_q(xt)
            q_t = F.normalize(q_t, p=2, dim=-1)
            read_val = torch.bmm(memory_state, q_t.unsqueeze(-1)).squeeze(-1)

            # Combine
            combined = torch.cat([xt, read_val], dim=-1)
            hidden = F.relu(self.fc_out1(combined))
            logits = self.fc_out2(hidden)

            out_logits.append(logits.unsqueeze(1))

        out_logits = torch.cat(out_logits, dim=1)
        return out_logits, memory_state
