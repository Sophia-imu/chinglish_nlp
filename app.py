# app.py
# ============================================================
# Chinglish NLP 演示
# Tab1 词向量调参 | Tab2 TextCNN 文本分类 | Tab3 多模型对照
# ============================================================

import os
import re
import string
import numpy as np
import pandas as pd
import streamlit as st

# ============ 全局配置（按需修改路径） ============
CSV_PATH = "chinglish_data.csv"          # Tab2/3：分类数据（text,label）
GENERAL_CORPUS = "general_corpus.txt"    # Tab1：通用英文语料（每行一句）
WORDS_PATH = "words.csv"                 # Tab1：词汇/对照语料（.csv 或 .txt）
W2V_MODEL_DIR = "."                      # 模型保存目录
MAX_LEN = 20
EMBED_DIM = 64
SEED = 42

np.random.seed(SEED)

# ============ 中文显示（Matplotlib） ============
import matplotlib
matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
matplotlib.rcParams["axes.unicode_minus"] = False

def safe_read_csv(path):
    """自动兼容 UTF-8 和 GBK 编码的 CSV。"""
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="gbk")


def safe_read_lines(path):
    """读取文本文件所有行，自动兼容 UTF-8 和 GBK。"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().splitlines()
    except UnicodeDecodeError:
        with open(path, "r", encoding="gbk") as f:
            return f.read().splitlines()


# ============ 工具函数 ============
def tokenize(text: str):
    text = str(text).lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    return [w for w in text.split() if w]


def model_filename(vector_size, window, negative, epochs):
    return os.path.join(
        W2V_MODEL_DIR,
        f"word2vec.model.d{vector_size}_w{window}_n{negative}_e{epochs}"
    )


def load_w2v_corpus(general_path, words_path):
    """合并 通用语料 + words 语料，返回句子列表（每个句子是词列表）。"""
    raw = []

    # 1) 通用英文语料
    if os.path.exists(general_path):
        for line in safe_read_lines(general_path):
            line = line.strip()
            if line:
                raw.append(line)
    else:
        print(f"[警告] 未找到通用语料：{general_path}")

    # 2) words 语料：兼容 .txt / .csv
    if os.path.exists(words_path):
        if words_path.endswith(".txt"):
            for line in safe_read_lines(words_path):
                line = line.strip()
                if line:
                    raw.append(line)
        else:
            df=safe_read_csv(words_path)
            text_cols = [
                c for c in df.columns
                if df[c].dtype == object and c.lower() not in ("label", "id", "index")
            ]
            for _, row in df.iterrows():
                for c in text_cols:
                    v = str(row[c]).strip()
                    if v and v.lower() != "nan":
                        raw.append(v)
    else:
        print(f"[警告] 未找到 words 语料：{words_path}")

        # ===== 新增：去重 =====
    seen = set()
    deduped = []
    for s in raw:
        key = s.strip().lower()
        if key not in seen:
            seen.add(key)
            deduped.append(s)

    sentences = [tokenize(t) for t in deduped]
    return [s for s in sentences if s]


# ============ Tab 1: 词向量调参 ============
def tab_word2vec():
    st.header("Tab 1 · 词向量调参")
    st.write("滑块调超参、查相似词、做词类比、看 PCA 2D 投影与参数差异。")

    # --- 超参数 ---
    with st.expander("⚙️ 超参数调节（改完点训练）", expanded=False):
        vector_size = st.slider("向量维度 vector_size", 50, 300, 100, 10)
        window = st.slider("窗口大小 window", 2, 10, 5, 1)
        negative = st.slider("负采样 negative", 1, 15, 5, 1)
        epochs = st.slider("训练轮数 epochs", 5, 100, 30, 5)

    topn = st.slider("相似词数量 (Top-N)", 3, 30, 10, 1)

    # --- 当前参数对应模型路径 ---
    model_path = model_filename(vector_size, window, negative, epochs)

    has_general = os.path.exists(GENERAL_CORPUS)
    has_words = os.path.exists(WORDS_PATH)

    # --- 模型不存在：训练 ---
    if not os.path.exists(model_path):
        st.warning(
            f"当前参数组合的模型不存在：{model_path}\n\n"
            f"通用语料 {GENERAL_CORPUS}：{'✓ 存在' if has_general else '✗ 不存在'}\n\n"
            f"words 语料 {WORDS_PATH}：{'✓ 存在' if has_words else '✗ 不存在'}"
        )

        if (has_general or has_words) and st.button("🚀 用当前超参数训练 Word2Vec"):
            with st.spinner("训练中..."):
                from gensim.models import Word2Vec
                sentences = load_w2v_corpus(GENERAL_CORPUS, WORDS_PATH)
                st.info(f"语料共 {len(sentences)} 条句子")

                if len(sentences) == 0:
                    st.error("语料为空，请检查两份数据文件。")
                else:
                    model = Word2Vec(
                        sentences,
                        vector_size=vector_size,
                        window=window,
                        negative=negative,
                        min_count=1,
                        workers=2,
                        seed=SEED,
                        epochs=epochs,
                    )
                    model.save(model_path)
                    st.success(f"已保存到 {model_path}，请刷新页面。")
        return

    # --- 加载模型 ---
    @st.cache_resource(show_spinner=False)
    def load_w2v(path):
        from gensim.models import Word2Vec
        return Word2Vec.load(path)

    model = load_w2v(model_path)

    # --- 相似词查询 ---
    query = st.text_input("请输入词汇（如 'add oil'、'good'）", "add oil")
    tokens = tokenize(query)
    if not tokens:
        st.info("请输入一个有效词汇。")
        return

    valid_tokens = [t for t in tokens if t in model.wv]
    if not valid_tokens:
        st.error(f"词 '{query}' 不在词表中。请换一个词试试。")
        return

    key = valid_tokens[0]
    sims = model.wv.most_similar(key, topn=topn)

    st.subheader(f"与 **{key}** 最相似的 {topn} 个词")
    st.dataframe(
        pd.DataFrame(sims, columns=["词", "相似度"]),
        use_container_width=True
    )

    # --- 显示语料中的对照行（仅在 .csv 时） ---
    if has_words and WORDS_PATH.endswith(".csv"):
        ref =safe_read_csv(WORDS_PATH)
        text_cols = [c for c in ref.columns if ref[c].dtype == object]
        if len(text_cols) >= 2:
            c_cn, c_en = text_cols[0], text_cols[1]
            hit = ref[
                ref[c_cn].astype(str).str.contains(key, case=False, na=False) |
                ref[c_en].astype(str).str.contains(key, case=False, na=False)
            ]
            if not hit.empty:
                st.write("**语料中的对照行：**")
                st.dataframe(hit, use_container_width=True)

    # --- 词类比计算 a - b + c = ? ---
    st.markdown("---")
    st.subheader("🧮 词类比计算 a - b + c = ?")
    col1, col2, col3 = st.columns(3)
    a = col1.text_input("a", "good")
    b = col2.text_input("b", "study")
    c = col3.text_input("c", "diligent")
    if st.button("计算类比"):
        try:
            res = model.wv.most_similar(
                positive=[tokenize(c)[0], tokenize(a)[0]],
                negative=[tokenize(b)[0]],
                topn=5,
            )
            st.write(f"**{a} - {b} + {c} =**")
            for w, s in res:
                st.write(f"- {w} （相似度 {s:.3f}）")
        except KeyError as e:
            st.error(f"词不在词表中：{e}")

    # --- PCA 2D 可视化 ---
    st.markdown("---")
    st.subheader("📉 参数变化带来的差异对比 · PCA 2D")
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt

    words = [key] + [w for w, _ in sims]
    vectors = np.array([model.wv[w] for w in words])
    pca = PCA(n_components=2, random_state=SEED)
    coords = pca.fit_transform(vectors)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(coords[1:, 0], coords[1:, 1], c="steelblue", s=80, label="相似词")
    ax.scatter(coords[0, 0], coords[0, 1], c="crimson", s=140, marker="*",
               label=f"查询词: {key}")
    for i, w in enumerate(words):
        ax.annotate(w, (coords[i, 0], coords[i, 1]),
                    fontsize=9, xytext=(4, 4), textcoords="offset points")
    ax.set_title(f"Word2Vec 向量 PCA 2D 投影 (Top-{topn})")
    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig)


# ============ 简易 TextCNN (PyTorch) ============
def build_vocab(texts, min_freq=1):
    from collections import Counter
    cnt = Counter()
    for t in texts:
        cnt.update(tokenize(t))
    vocab = {"<pad>": 0, "<unk>": 1}
    for w, c in cnt.items():
        if c >= min_freq:
            vocab[w] = len(vocab)
    return vocab


def encode(texts, vocab, max_len=MAX_LEN):
    ids = []
    for t in texts:
        seq = [vocab.get(w, vocab["<unk>"]) for w in tokenize(t)][:max_len]
        seq += [vocab["<pad>"]] * (max_len - len(seq))
        ids.append(seq)
    return np.array(ids, dtype=np.int64)


def train_textcnn(texts, labels, epochs=8, lr=1e-3):
    import torch
    import torch.nn as nn

    class TextCNN(nn.Module):
        def __init__(self, vocab_size, embed_dim=EMBED_DIM, num_class=2):
            super().__init__()
            self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            self.convs = nn.ModuleList([
                nn.Conv1d(embed_dim, 64, k) for k in (2, 3, 4)
            ])
            self.dropout = nn.Dropout(0.3)
            self.fc = nn.Linear(64 * 3, num_class)

        def forward(self, x):
            e = self.emb(x).transpose(1, 2)
            feats = [torch.relu(c(e)).max(dim=2).values for c in self.convs]
            h = torch.cat(feats, dim=1)
            return self.fc(self.dropout(h))

    vocab = build_vocab(texts)
    X = torch.tensor(encode(texts, vocab))
    y = torch.tensor(labels, dtype=torch.long)

    model = TextCNN(len(vocab))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()

    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        out = model(X)
        loss = lossf(out, y)
        loss.backward()
        opt.step()

    model.eval()
    return model, vocab


# ============ Tab 2: CNN 文本分类 ============
def tab_cnn():
    st.header("Tab 2 · TextCNN 文本分类")
    st.write("读取本地 `chinglish_data.csv`（列：`text`, `label`），训练一个简易 TextCNN。")

    if not os.path.exists(CSV_PATH):
        st.error(f"未找到数据文件：{CSV_PATH}")
        return

    @st.cache_data(show_spinner=False)
    def load_csv(path):
        return safe_read_csv(path)

    df = load_csv(CSV_PATH)
    if "text" not in df.columns or "label" not in df.columns:
        st.error("CSV 必须包含 `text` 和 `label` 两列。")
        return

    st.write(f"数据量：{len(df)} 条")
    st.dataframe(df.head(), use_container_width=True)

    unique = list(pd.unique(df["label"]))
    if len(unique) != 2:
        st.error(f"label 列需要恰好 2 个类别，当前为: {unique}")
        return
    st.write(f"类别：{unique}")

    if st.button("训练 / 重新训练 TextCNN") or "cnn_model" not in st.session_state:
        with st.spinner("训练 TextCNN 中..."):
            texts = df["text"].astype(str).tolist()
            labels = [0 if str(l) == str(unique[0]) else 1 for l in df["label"]]
            model, vocab = train_textcnn(texts, labels)
            st.session_state["cnn_model"] = model
            st.session_state["cnn_vocab"] = vocab
            st.session_state["cnn_classes"] = unique
        st.success("训练完成！")

    model = st.session_state["cnn_model"]
    vocab = st.session_state["cnn_vocab"]
    classes = st.session_state["cnn_classes"]

    sentence = st.text_input("输入一句英文：", "I very like this book")
    if st.button("预测 (TextCNN)"):
        import torch
        x = torch.tensor(encode([sentence], vocab))
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

        chinglish_idx = 1
        for i, c in enumerate(classes):
            cs = str(c).lower()
            if "chinglish" in cs or "中式" in cs or cs in ("1", "yes", "true"):
                chinglish_idx = i
                break
        native_idx = 1 - chinglish_idx

        st.subheader("预测结果")
        st.write("**中式英语** 概率")
        st.progress(float(probs[chinglish_idx]))
        st.write(f"{probs[chinglish_idx]*100:.2f}%")

        st.write("**地道英语** 概率")
        st.progress(float(probs[native_idx]))
        st.write(f"{probs[native_idx]*100:.2f}%")


# ============ Tab 3: 多模型对照 ============
def tab_compare():
    st.header("Tab 3 · 多模型对照（NB / SVM / TextCNN）")
    st.write("读取同一 CSV，分别训练朴素贝叶斯、SVM、TextCNN。")

    if not os.path.exists(CSV_PATH):
        st.error(f"未找到数据文件：{CSV_PATH}")
        return

    @st.cache_data(show_spinner=False)
    def load_csv2(path):
        return safe_read_csv(path)

    df = load_csv2(CSV_PATH)
    if "text" not in df.columns or "label" not in df.columns:
        st.error("CSV 必须包含 `text` 和 `label` 两列。")
        return

    unique = list(pd.unique(df["label"]))
    if len(unique) != 2:
        st.error(f"label 列需要恰好 2 个类别，当前为: {unique}")
        return

    texts = df["text"].astype(str).tolist()
    y = np.array([0 if str(l) == str(unique[0]) else 1 for l in df["label"]])

    @st.cache_resource(show_spinner=False)
    def train_all(texts, y):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.svm import LinearSVC
        from sklearn.calibration import CalibratedClassifierCV

        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        X = vec.fit_transform(texts)

        nb = MultinomialNB().fit(X, y)
        svm = CalibratedClassifierCV(LinearSVC(), cv=3).fit(X, y)

        cnn_model, vocab = train_textcnn(texts, list(y))
        return vec, nb, svm, cnn_model, vocab

    with st.spinner("训练三个模型中..."):
        vec, nb, svm, cnn_model, cnn_vocab = train_all(tuple(texts), tuple(y))

    model_name = st.selectbox("选择模型", ["朴素贝叶斯 (NB)", "SVM", "TextCNN"])
    sentence = st.text_input("输入英文句子", "I very like this food")

    if st.button("预测 (对比)"):
        if model_name.startswith("朴素"):
            probs = nb.predict_proba(vec.transform([sentence]))[0]
        elif model_name == "SVM":
            probs = svm.predict_proba(vec.transform([sentence]))[0]
        else:
            import torch
            x = torch.tensor(encode([sentence], cnn_vocab))
            with torch.no_grad():
                logits = cnn_model(x)
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

        pred = int(np.argmax(probs))
        conf = float(probs[pred])
        label_name = {0: str(unique[0]), 1: str(unique[1])}

        st.success(f"【{model_name}】预测：**{label_name[pred]}**  置信度：**{conf*100:.2f}%**")
        st.write("各类别概率：")
        for i, u in enumerate(unique):
            st.write(f"- {u}: {probs[i]*100:.2f}%")
            st.progress(float(probs[i]))


# ============ 主入口 ============
def main():
    st.set_page_config(page_title="Chinglish NLP Demo", layout="wide")
    st.title("Chinglish NLP 演示 · 词向量 / TextCNN / 多模型对照")

    tab1, tab2, tab3 = st.tabs([
        "Tab1 词向量调参",
        "Tab2 CNN 文本分类",
        "Tab3 多模型对照",
    ])

    with tab1:
        tab_word2vec()
    with tab2:
        tab_cnn()
    with tab3:
        tab_compare()


if __name__ == "__main__":
    main()