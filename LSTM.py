import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from d2l import torch as d2l
import re
from collections import defaultdict

# -------------------------- 1. 数据预处理 --------------------------
def preprocess_text(raw_text):
    """
    文本清洗： lowercase + 移除特殊字符 + 保留字母、空格、基本标点
    """
    # 转为小写，移除非字母、空格、标点（. , ! ? ;）的字符
    text = raw_text.lower()
    text = re.sub(r"[^a-z\s\.,!?;]", "", text)
    # 合并多个空格为单个空格
    text = re.sub(r"\s+", " ", text)
    return text

def build_vocab(text, tokenize="char"):
    """
    构建词表
    tokenize: "char"（字符级）或 "word"（单词级），实验用字符级
    """
    # 词元化（字符级：每个字符作为一个词元）
    if tokenize == "char":
        tokens = list(text)  # 按字符分割
    else:
        tokens = text.split()  # 按单词分割
    
    # 统计词频，构建char_to_idx和idx_to_char
    vocab = defaultdict(int)
    for token in tokens:
        vocab[token] += 1
    # 按词频排序
    vocab = sorted(vocab.items(), key=lambda x: x[1], reverse=True)
    # 构建映射：字符→索引，索引→字符
    char_to_idx = {char: idx for idx, (char, _) in enumerate(vocab)}
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}
    return char_to_idx, idx_to_char, len(char_to_idx)

def text_to_indices(text, char_to_idx):
    """将文本转为索引序列"""
    return [char_to_idx[char] for char in text if char in char_to_idx]

def generate_seq_data(data_indices, seq_len, batch_size):
    """
    生成批量序列数据
    输入：data_indices（文本索引序列）、seq_len（序列长度）、batch_size（批量大小）
    输出：X（输入序列，shape=(batch_size, seq_len)）、Y（目标序列，shape=(batch_size, seq_len)）
    """
    # 计算总步数：每个序列对应一个目标（X[i]→Y[i]，Y是X的偏移1位）
    total_steps = len(data_indices) - seq_len
    # 随机采样batch_size个起始索引
    indices = torch.randint(0, total_steps, (batch_size,))
    
    # 生成输入X和目标Y
    X = torch.tensor([data_indices[i:i+seq_len] for i in indices], dtype=torch.long)
    Y = torch.tensor([data_indices[i+1:i+seq_len+1] for i in indices], dtype=torch.long)
    return X, Y

# 加载并预处理数据集
def load_time_machine_data(seq_len=35, batch_size=32):
    """
    加载The Time Machine数据集，返回批量数据迭代器+词表
    seq_len：序列长度35，batch_size：批量大小32
    """
    # 1. 读取数据集
    fname = "timemachine.txt"
    with open(fname, 'r', encoding='utf-8') as f:
        raw_text = f.read()
    
    # 2. 文本预处理
    text = preprocess_text(raw_text)
    print(f"预处理后文本长度：{len(text)} 字符")
    print(f"预处理后前50字符：{text[:50]}")
    
    # 3. 构建词表
    char_to_idx, idx_to_char, vocab_size = build_vocab(text, tokenize="char")
    print(f"词表大小（不同字符数）：{vocab_size}")
    
    # 4. 文本转索引
    data_indices = text_to_indices(text, char_to_idx)
    print(f"索引序列长度：{len(data_indices)}")
    
    # 5. 生成批量数据迭代器
    def data_iter():
        while True:
            X, Y = generate_seq_data(data_indices, seq_len, batch_size)
            yield X, Y
    
    return data_iter(), char_to_idx, idx_to_char, vocab_size

# -------------------------- 2. LSTM模型定义 --------------------------
class LSTMGenerator(nn.Module):
    """
    LSTM句子生成模型
    结构：嵌入层→LSTM层→全连接层（映射到词表大小）
    """
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, num_layers=2, dropout=0.2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)  # 字符嵌入层
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True  # 输入格式：(batch_size, seq_len, embed_dim)
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)  # 输出层：映射到词表大小
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

    def forward(self, x, hidden=None):
        """前向传播：x为输入序列，hidden为LSTM隐藏状态"""
        # 嵌入层：(batch_size, seq_len) → (batch_size, seq_len, embed_dim)
        embed = self.embed(x)
        # LSTM层：输出→(batch_size, seq_len, hidden_dim)，hidden→(num_layers, batch_size, hidden_dim)
        lstm_out, hidden = self.lstm(embed, hidden)
        # 全连接层：(batch_size, seq_len, hidden_dim) → (batch_size, seq_len, vocab_size)
        output = self.fc(lstm_out)
        return output, hidden

    def generate_text(self, prompt, char_to_idx, idx_to_char, gen_len=200, temperature=0.7):
        """
        文本生成（教材8.4.3节：基于RNN的预测，适配LSTM）
        prompt：提示词（实验要求的10个提示词），gen_len：生成文本长度，temperature：随机性控制
        """
        self.eval()  # 切换到评估模式
        with torch.no_grad():
            # 1. 提示词转索引
            prompt_indices = [char_to_idx[char] for char in prompt.lower() if char in char_to_idx]
            if not prompt_indices:
                return f"提示词[{prompt}]中无有效字符！"
            
            # 2. 初始化输入和隐藏状态
            input_seq = torch.tensor([prompt_indices], dtype=torch.long)  # (1, seq_len_prompt)
            hidden = None  # LSTM隐藏状态初始化为None（自动初始化）
            generated = list(prompt)  # 保存生成结果（包含提示词）
            
            # 3. 逐字符生成
            for _ in range(gen_len):
                output, hidden = self.forward(input_seq, hidden)
                # 取最后一个字符的预测结果，应用温度调节
                last_output = output[:, -1, :] / temperature  # (1, vocab_size)
                prob = torch.softmax(last_output, dim=1)
                next_idx = torch.multinomial(prob, num_samples=1).item()  # 采样下一个字符索引
                
                # 更新输入和生成结果
                next_char = idx_to_char[next_idx]
                generated.append(next_char)
                input_seq = torch.tensor([[next_idx]], dtype=torch.long)  # 下一轮输入为当前生成字符
            
            return "".join(generated)

# -------------------------- 3. 模型训练 --------------------------
def train_lstm(net, data_iter, char_to_idx, idx_to_char, vocab_size, 
               epochs=50, lr=1e-3, device=None, grad_clip=1.0):
    """
    训练LSTM模型（教材8.5.6节：训练流程，含梯度裁剪）
    grad_clip：梯度裁剪阈值（防止梯度爆炸，教材8.5.5节）
    """
    # 设备选择：自动选择GPU/CPU
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)
    
    # 损失函数：交叉熵损失
    criterion = nn.CrossEntropyLoss()
    # 优化器：Adam
    optimizer = optim.Adam(net.parameters(), lr=lr)
    # 学习率调度器：损失不下降时衰减学习率
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    
    # 记录困惑度
    perplexity_list = []
    
    net.train()  # 切换到训练模式
    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 100  # 每轮训练的批次数（根据数据量调整，避免训练过久）
        
        for _ in range(num_batches):
            # 1. 获取批量数据并移动到设备
            X, Y = next(data_iter)
            X, Y = X.to(device), Y.to(device)
            
            # 2. 前向传播
            output, _ = net(X)  # output: (batch_size, seq_len, vocab_size)
            
            # 3. 计算损失（CrossEntropyLoss要求input=(N, C), target=(N)）
            output = output.reshape(-1, vocab_size)  # (batch_size*seq_len, vocab_size)
            Y = Y.reshape(-1)  # (batch_size*seq_len)
            loss = criterion(output, Y)
            
            # 4. 反向传播与优化
            optimizer.zero_grad()  # 清空梯度
            loss.backward()  # 反向传播
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=grad_clip)
            optimizer.step()  # 更新参数
            
            total_loss += loss.item()
        
        # 5. 计算每轮的平均损失和困惑度
        avg_loss = total_loss / num_batches
        perplexity = torch.exp(torch.tensor(avg_loss)).item()  # 困惑度=exp(平均损失)
        perplexity_list.append(perplexity)
        # 学习率调度（根据平均损失调整）
        scheduler.step(avg_loss)
        
        # 6. 打印训练信息
        print(f"Epoch [{epoch+1}/{epochs}] | "
              f"Avg Loss: {avg_loss:.4f} | "
              f"Perplexity: {perplexity:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")
    
    # 绘制困惑度曲线
    plt.figure(figsize=(10, 4))
    plt.plot(range(1, epochs+1), perplexity_list, label="Training Perplexity")
    plt.xlabel("Epoch")
    plt.ylabel("Perplexity")
    plt.title("LSTM Training Perplexity Curve (The Time Machine)")
    plt.legend()
    plt.grid(True)
    plt.savefig("lstm_perplexity.png") 
    print("困惑度曲线已保存为：lstm_perplexity.png")
    
    # 保存模型（教材5.5节：读写文件）
    torch.save({
        "model_state_dict": net.state_dict(),
        "char_to_idx": char_to_idx,
        "idx_to_char": idx_to_char,
        "vocab_size": vocab_size
    }, "lstm_text_generator.pth")
    print("模型已保存为：lstm_text_generator.pth")
    
    return net

# -------------------------- 4. 实验测试：生成10个提示词的文本 --------------------------
def test_prompts(model_path="lstm_text_generator.pth", gen_len=200):
    """
    加载训练好的模型，测试实验四要求的10个提示词
    提示词列表：实验四“实验要求”部分指定的10个提示词
    """
    # 1. 加载模型和词表
    checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
    vocab_size = checkpoint["vocab_size"]
    char_to_idx = checkpoint["char_to_idx"]
    idx_to_char = checkpoint["idx_to_char"]
    
    # 2. 初始化模型并加载参数
    model = LSTMGenerator(
        vocab_size=vocab_size,
        embed_dim=128,
        hidden_dim=256,
        num_layers=2,
        dropout=0.2
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    
    # 3. 实验四要求的10个提示词
    experiment_prompts = [
        "time traveller",
        "traveller",
        "the time traveller says that",
        "when the time traveller returns to the garden",
        "the time traveller begins learning the language",
        "the time traveller determines that",
        "the time traveller knows he will have to stop",
        "when he wakes up",
        "the time traveller finds himself",
        "the time traveller tells the narrator to wait for him"
    ]
    
    # 4. 生成文本并打印
    print("\n" + "="*50)
    print("实验四：10个提示词的文本生成结果")
    print("="*50 + "\n")
    
    results = {}
    for i, prompt in enumerate(experiment_prompts, 1):
        generated_text = model.generate_text(
            prompt=prompt,
            char_to_idx=char_to_idx,
            idx_to_char=idx_to_char,
            gen_len=gen_len,
            temperature=0.7  # 温度0.7：平衡随机性和连贯性
        )
        results[prompt] = generated_text
        print(f"【提示词{i}】：{prompt}")
        print(f"【生成结果】：{generated_text}\n")
        print("-"*80 + "\n")
    
    # 保存生成结果到文件（实验报告可用）
    with open("lstm_generated_results.txt", "w", encoding="utf-8") as f:
        for prompt, text in results.items():
            f.write(f"提示词：{prompt}\n")
            f.write(f"生成结果：{text}\n")
            f.write("-"*80 + "\n\n")
    print("所有生成结果已保存为：lstm_generated_results.txt")
    return results

# -------------------------- 5. 主函数（实验执行流程） --------------------------
if __name__ == "__main__":
    # 超参数设置
    seq_len = 35        # 序列长度
    batch_size = 32     # 批量大小
    embed_dim = 128     # 嵌入维度
    hidden_dim = 256    # LSTM隐藏层维度
    num_layers = 2      # LSTM层数
    dropout = 0.2       #  dropout概率
    epochs = 50         # 训练轮数
    lr = 1e-3           # 学习率
    gen_len = 200       # 生成文本长度
    
    # 步骤1：加载数据集
    print("="*50)
    print("步骤1：加载The Time Machine数据集")
    print("="*50)
    data_iter, char_to_idx, idx_to_char, vocab_size = load_time_machine_data(
        seq_len=seq_len,
        batch_size=batch_size
    )
    
    # 步骤2：初始化模型
    print("\n" + "="*50)
    print("步骤2：初始化LSTM模型")
    print("="*50)
    net = LSTMGenerator(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout
    )
    print(f"模型结构：\n{net}")
    
    # 步骤3：训练模型
    print("\n" + "="*50)
    print("步骤3：训练LSTM模型（首次运行约30分钟，GPU加速更快）")
    print("="*50)
    net = train_lstm(
        net=net,
        data_iter=data_iter,
        char_to_idx=char_to_idx,
        idx_to_char=idx_to_char,
        vocab_size=vocab_size,
        epochs=epochs,
        lr=lr
    )
    
    # 步骤4：测试10个提示词
    print("\n" + "="*50)
    print("步骤4：测试实验四要求的10个提示词")
    print("="*50)
    results = test_prompts(
        model_path="lstm_text_generator.pth",
        gen_len=gen_len
    )
