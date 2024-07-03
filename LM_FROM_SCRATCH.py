import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)

batch_size = 16
block_size = 64
max_iter = 5000
eval_interval = 100
learning_rate = 1e-3
device = 'cude' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embed =  64
n_head = 4
n_layer = 4
dropout = 0.0



with open('input.txt','r',encoding='utf-8') as f:
  text=f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }

encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

data = torch.tensor(encode(text), dtype=torch.long)

# Let's now split up the data into train and validation sets
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y= x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
  out = {}
  model.eval()
  for split in ['train', 'val']:
    losses =torch.zeros(eval_iters)
    for k in range(eval_iters):
        x, y = get_batch(split)
        logits, loss = model(x, y)
        losses[k]=loss.item()
    out[split]=losses.mean()
  model.train()
  return out
 

class Head(nn.Module):
  """one head attention"""
  def __init__(self,head_size):
    super().__init__()
    self.key = nn.Linear(n_embed, head_size, bias = False)
    self.query = nn.Linear(n_embed, head_size, bias = False)
    self.value = nn.Linear(n_embed, head_size, bias = False)
    self.register_buffer('tril',torch.tril(torch.ones(block_size, block_size)))
    self.dropout = nn.Dropout(dropout)


  def forward(self, x):
    B, T, C =x.shape
    k = self.key(x)
    q = self.query(x)
    wei = q @ k.transpose(-2,-1) * C ** -0.5
    wei = wei.masked_fill(self.tril[:T,:T] == 0,float('-inf'))
    wei = F.softmax(wei,dim=-1)
    wei = self.dropout(wei)
    v = self.value(x)
    out = wei @ v
    return out 



class MultiHeadAttention(nn.Module):
  """  multiple head in parallel """

  def __init__(self,nums_head,head_size):
    super().__init__()
    self.heads = nn.ModuleList([Head(head_size) for _ in range(nums_head)])
    self.proj = nn.Linear(n_embed, n_embed)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    out = torch.cat([h(x) for h in self.heads], dim=-1)
    out = self.dropout(self.proj(out))
    return out


class FeedForward(nn.Module):
  """ a simple linear layer followed by a non-linearity
      After self attention when each token got the info from others
      now it's chance to think on it that's what feedord for 
    
   """
  def __init__(self,n_embed):
    super().__init__()
    self.net =  nn.Sequential(
      nn.Linear(n_embed , 4 * n_embed),
      nn.ReLU(),
      nn.Linear(4 * n_embed, n_embed),
      nn.Dropout(dropout)
    )

  def forward(self, x):
    return self.net(x)


class Block(nn.Module):
  """ transformer block: communication followed by comutation """
  def __init__(self, n_embed, n_head):
    super().__init__()

    head_size = n_embed // n_head

    self.sa = MultiHeadAttention(n_head, head_size)
    self.ffwd = FeedForward(n_embed)

    ''''
      mean = X.mean(dim=-1, keepdim=True)
      variance = X.var(dim=-1, keepdim=True, unbiased=False) + epsilon
      X_norm = (X - mean) / torch.sqrt(variance)
      output = gamma * X_norm + beta
      
      ** this is pertoken normalization
    '''
    self.ln1 = nn.LayerNorm(n_embed)
    self.ln2 = nn.LayerNorm(n_embed)

  def forward(self,x):
    xatt = x + self.sa(self.ln1(x))
    xout = x + self.ffwd(self.ln2(xatt))
    return xout



class BiagramLanguageModel(nn.Module):

  def __init__(self):
    super().__init__()
    self.token_embedding_table = nn.Embedding(vocab_size,n_embed)
    self.position_embed_table = nn.Embedding(block_size,n_embed)
    
    self.blocks = nn.Sequential(*[Block(n_embed, n_head) for _ in range(n_layer)])
    self.ln_f = nn.LayerNorm(n_embed)
    self.lm_head = nn.Linear(n_embed, vocab_size)

  def forward(self,idx,targets=None):
    B, T = idx.shape

    tok_emb = self.token_embedding_table(idx)
    pos_emb = self.position_embed_table(torch.arange(T, device = device))
    x = tok_emb + pos_emb
    x = self.blocks(x)
    x = self.ln_f(x)

    logits=self.lm_head(x) #(B,T,C)

    '''
      WHERE B : BATCH SIZE  , T : BLOCK , C : VOCAB SIZE

      so this is used for prediction of next token

    '''
    if targets is None:
      loss=None
    else:
      B,T,C = logits.shape
      logits=logits.view(B*T,C)
      targets=targets.view(B*T)
      loss= F.cross_entropy(logits,targets)

    return logits,loss

  def generate(self,idx,max_new_tokens):

    for _ in range(max_new_tokens):
      idx_cond = idx[:,-block_size:]
      logits, loss = self(idx_cond)  #loss will be ignored
      logits = logits[:,-1,:]
      probs = F.softmax(logits,dim=-1)
      idx_next =torch.multinomial(probs,num_samples=1) #(B,1) for each batch 1 samples
      idx=torch.cat((idx,idx_next),dim=1) #(B,T+1)

    return idx



model = BiagramLanguageModel()
m=model.to(device)


optimizer=torch.optim.AdamW(m.parameters(),lr=learning_rate)

for step in range(max_iter):
  if step % eval_interval == 0 or step == max_iter-1 :
     losses = estimate_loss()
     print(f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
     
  xb, yb = get_batch('train')
  logits, loss = m(xb,yb)
  optimizer.zero_grad(set_to_none = True)
  loss.backward()
  optimizer.step()

print(f'final loss : {loss.item():4f}')

print(decode(m.generate(idx = torch.zeros((1, 1), dtype=torch.long), max_new_tokens=500)[0].tolist()))

