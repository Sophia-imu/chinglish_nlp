# app.py
import os
import re
import string
from collections import Counter
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ============ 全局配置 ============
CSV_PATH = "chinglish_data.csv"
GENERAL_CORPUS = "general_corpus.txt"
WORDS_PATH = "words.csv"
W2V_MODEL_DIR = "w2v_models"
SEED = 42
np.random.seed(SEED)


# ============ 工具函数 ============
def safe_read_csv(path):
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="gbk")


def safe_read_lines(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().splitlines()
    except UnicodeDecodeError:
        with open(path, "r", encoding="gbk") as f:
            return f.read().splitlines()


def tokenize(text):
    text = str(text).lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    return [w for w in text.split() if w]


def load_w2v_corpus():
    raw = []

    if os.path.exists(GENERAL_CORPUS):
        for line in safe_read_lines(GENERAL_CORPUS):
            line = line.strip()
            if line:
                raw.append(line)

    if os.path.exists(WORDS_PATH):
        if WORDS_PATH.endswith(".txt"):
            for line in safe_read_lines(WORDS_PATH):
                line = line.strip()
                if line:
                    raw.append(line)
        else:
            df = safe_read_csv(WORDS_PATH)
            text_cols = list(df.columns)     # 👈 这行保留
            for _, row in df.iterrows():
                for c in text_cols:
                    v = str(row[c]).strip()
                    if v and v.lower() != "nan":
                        raw.append(v)

    seen = set()
    deduped = []
    for s in raw:
        key = s.strip().lower()
        if key not in seen:
            seen.add(key)
            deduped.append(s)

    sentences = [tokenize(t) for t in deduped]
    return [s for s in sentences if s]


# ==================== Tab 1: 词向量调参 ====================
def tab_word2vec():
    st.header("Tab 1 · 词向量调参可视化")
    st.caption("玩法：滑块调超参数 → 查相似词 → 词类比 → PCA 2D → 参数对比")

    with st.expander("⚙️ 超参数调节（改完点训练）", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            vector_size = st.slider("向量维度 vector_size", 50, 300, 100, 50)
            window = st.slider("窗口大小 window", 2, 10, 8)
        with c2:
            min_count = st.slider("最低词频 min_count", 1, 5, 1)
            epochs = st.slider("训练轮数 epochs", 10, 200, 110, 10)
        sg_choice = st.radio("训练算法", ["Skip-Gram (sg=1)", "CBOW (sg=0)"], index=0, horizontal=True)
        sg_val = 1 if sg_choice.startswith("Skip") else 0

    model_name = f"w2v_d{vector_size}_w{window}_mc{min_count}_e{epochs}_sg{sg_val}.model"
    os.makedirs(W2V_MODEL_DIR, exist_ok=True)
    model_path = os.path.join(W2V_MODEL_DIR, model_name)

    has_general = os.path.exists(GENERAL_CORPUS)
    has_words = os.path.exists(WORDS_PATH)

    c1, c2 = st.columns([1, 2])
    with c1:
        train_btn = st.button("🚀 用当前超参数训练 Word2Vec")
    with c2:
        if os.path.exists(model_path):
            st.success(f"✅ 已存在：{model_name}")
        else:
            st.info(f"通用语料 {'✓' if has_general else '✗'} | words {'✓' if has_words else '✗'}")

    if train_btn:
        if not (has_general or has_words):
            st.error("找不到语料文件。")
        else:
            with st.spinner("训练中..."):
                from gensim.models import Word2Vec
                sentences = load_w2v_corpus()
                if len(sentences) == 0:
                    st.error("语料为空。")
                else:
                    st.write(f"语料共 {len(sentences)} 条句子")
                    model = Word2Vec(
                        sentences,
                        vector_size=vector_size,
                        window=window,
                        min_count=min_count,
                        sg=sg_val,
                        negative=10,
                        epochs=epochs,
                        workers=4,
                        seed=SEED,
                    )
                    model.save(model_path)
                    st.success(f"已保存：{model_name}")
                    st.rerun()

    available_models = []
    if os.path.isdir(W2V_MODEL_DIR):
        available_models = sorted([f for f in os.listdir(W2V_MODEL_DIR) if f.endswith(".model")])

    if not os.path.exists(model_path):
        st.warning("当前参数组合的模型不存在，请点上方训练按钮。")
        return

    @st.cache_resource(show_spinner=False)
    def load_w2v(path):
        from gensim.models import Word2Vec
        return Word2Vec.load(path)

    model = load_w2v(model_path)

    st.markdown("---")
    topn = st.slider("相似词数量 Top-N", 1, 30, 5)
    query = st.text_input("请输入词汇（如 'add oil'、'good'）", "add oil")

    tokens = tokenize(query)
    valid_tokens = [t for t in tokens if t in model.wv]

    if not query.strip():
        st.warning("⚠️ 请输入词汇，不能为空！")
    elif not valid_tokens:
        st.error(f"'{query}' 不在词表中。")
    else:
        key = valid_tokens[0]
        sims = model.wv.most_similar(key, topn=topn)
        st.subheader(f"与 **{key}** 最相似的 {topn} 个词")
        st.dataframe(pd.DataFrame(sims, columns=["词", "相似度"]), use_container_width=True)

        if has_words and WORDS_PATH.endswith(".csv"):
            ref = safe_read_csv(WORDS_PATH)
            text_cols = [c for c in ref.columns if ref[c].dtype == object]
            if len(text_cols) >= 2:
                c1, c2 = text_cols[0], text_cols[1]
                hit = ref[
                    ref[c1].astype(str).str.contains(key, case=False, na=False)
                    | ref[c2].astype(str).str.contains(key, case=False, na=False)
                ]
                if not hit.empty:
                    st.write("**语料中的对照行：**")
                    st.dataframe(hit, use_container_width=True)

        from sklearn.decomposition import PCA
        words = [key] + [w for w, _ in sims]
        vectors = np.array([model.wv[w] for w in words])
        pca = PCA(n_components=2, random_state=SEED)
        coords = pca.fit_transform(vectors)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(coords[1:, 0], coords[1:, 1], c="steelblue", s=80, label="Similar Words")
        ax.scatter(coords[0, 0], coords[0, 1], c="crimson", s=150, marker="*", label=f"Query:{key}")
        for i, w in enumerate(words):
            ax.annotate(w, (coords[i, 0], coords[i, 1]), fontsize=9,
                        xytext=(4, 4), textcoords="offset points")
        ax.set_title(f"PCA 2D Projection Top-{topn}")
        ax.legend()
        ax.grid(alpha=0.3)
        st.pyplot(fig)

    st.markdown("---")
    st.subheader("🔢 词类比计算 a - b + c = ?")
    c1, c2, c3 = st.columns(3)
    a = c1.text_input("a", "add")
    b = c2.text_input("b", "oil")
    c = c3.text_input("c", "study")
    if st.button("计算类比"):
        try:
            result = model.wv.most_similar(positive=[a, c], negative=[b], topn=5)
            st.write(f"**{a} - {b} + {c} = ?**")
            for w, s in result:
                st.write(f"- {w}（相似度 {s:.3f}）")
        except KeyError as e:
            st.error(f"词不存在：{e}")

    st.markdown("---")
    st.subheader("📊 参数变化带来的差异对比")
    if len(available_models) < 2:
        st.info("目前只有一套参数模型。切换超参并训练第二套后再回来对比。")
    else:
        @st.cache_resource(show_spinner=False)
        def load_all(paths):
            from gensim.models import Word2Vec
            return [Word2Vec.load(p) for p in paths]

        paths = [os.path.join(W2V_MODEL_DIR, f) for f in available_models]
        models = load_all(paths)
        compare_word = st.text_input("对比查询词", "good")
        rows = []
        for fname, m in zip(available_models, models):
            if compare_word in m.wv:
                top = m.wv.most_similar(compare_word, topn=5)
                rows.append({
                    "模型": fname.replace(".model", ""),
                    "Top1": top[0][0], "Top2": top[1][0], "Top3": top[2][0],
                    "Top4": top[3][0], "Top5": top[4][0],
                })
            else:
                rows.append({"模型": fname.replace(".model", ""),
                             "Top1": "N/A", "Top2": "", "Top3": "", "Top4": "", "Top5": ""})
        st.dataframe(pd.DataFrame(rows), use_container_width=True)


# ==================== CNN 卷积池化可视化 ====================
def simulate_cnn(embed_matrix, kernel_sizes=(2, 3, 4, 5), num_filters=4, seed=42):
    """纯 numpy 模拟 1D 卷积 + 全局最大池化。"""
    seq_len, embed_dim = embed_matrix.shape
    results = {}
    for ks in kernel_sizes:
        if seq_len < ks:
            continue
        rng = np.random.RandomState(seed + ks)
        kernels = rng.randn(num_filters, ks, embed_dim) * 0.5
        conv_outs = np.zeros((num_filters, seq_len - ks + 1))
        for k in range(num_filters):
            for i in range(seq_len - ks + 1):
                window = embed_matrix[i:i + ks]
                conv_outs[k, i] = float(np.sum(window * kernels[k]))
        pooled = conv_outs.max(axis=1)
        results[ks] = {"kernels": kernels, "conv_outs": conv_outs, "pooled": pooled}
    return results


def draw_cnn_visual(embed_matrix, tokens, results, ks_show):
    seq_len, embed_dim = embed_matrix.shape
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ① 词向量矩阵
    ax = axes[0]
    im = ax.imshow(embed_matrix, aspect="auto", cmap="viridis")
    ax.set_title(f"① Embedding Matrix ({seq_len}×{embed_dim})")
    ax.set_yticks(range(seq_len))
    ax.set_yticklabels(tokens, fontsize=9)
    ax.set_xlabel("Embedding Dimension")
    plt.colorbar(im, ax=ax, fraction=0.04)

    # ② 卷积核滑动激活值
    ax = axes[1]
    if ks_show in results:
        conv = results[ks_show]["conv_outs"]
        im = ax.imshow(conv, aspect="auto", cmap="coolwarm")
        ax.set_title(f"② Conv Activation (kernel={ks_show})")
        ax.set_xticks(range(conv.shape[1]))
        ax.set_xticklabels([f"p{i}" for i in range(conv.shape[1])], rotation=45, fontsize=8)
        ax.set_yticks(range(conv.shape[0]))
        ax.set_yticklabels([f"filter{i+1}" for i in range(conv.shape[0])])
        ax.set_xlabel("Sliding Position")
        plt.colorbar(im, ax=ax, fraction=0.04)

    # ③ Max Pooling 结果
    ax = axes[2]
    if ks_show in results:
        pooled = results[ks_show]["pooled"]
        bars = ax.bar(range(len(pooled)), pooled, color="steelblue")
        for i, p in enumerate(pooled):
            ax.text(i, p, f"{p:.2f}", ha="center", va="bottom", fontsize=9)
        ax.set_title(f"③ Max Pooling Reault (kernel={ks_show})")
        ax.set_xticks(range(len(pooled)))
        ax.set_xticklabels([f"filter{i+1}" for i in range(len(pooled))])
        ax.set_ylabel("Max Activation")
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    return fig


def draw_conv_sliding_anim(embed_matrix, tokens, kernel_size):
    """画卷积核在词向量矩阵上滑动的分步示意图。"""
    seq_len, embed_dim = embed_matrix.shape
    positions = seq_len - kernel_size + 1
    if positions <= 0:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "Sentence too short", ha="center")
        return fig

    n_cols = min(positions, 4)
    n_rows = (positions + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 2.5 * n_rows))
    axes = np.array(axes).reshape(-1)

    for i in range(positions):
        ax = axes[i]
        ax.imshow(embed_matrix, aspect="auto", cmap="viridis", alpha=0.35)
        rect = Rectangle((-0.5, i - 0.5), embed_dim, kernel_size,
                         linewidth=2.5, edgecolor="red", facecolor="none")
        ax.add_patch(rect)
        cover = tokens[i:i + kernel_size]
        ax.set_title(f"Pos {i}: [{' '.join(cover)}]", fontsize=10)
        ax.set_yticks(range(seq_len))
        ax.set_yticklabels(tokens, fontsize=8)
        ax.set_xticks([])

    for j in range(positions, len(axes)):
        axes[j].axis("off")
    plt.tight_layout()
    return fig


def tab_cnn():
    st.header("Tab 2 · TextCNN 文本分类可视化")
    st.caption("玩法：训练分类器 → 输出概率 → 词向量矩阵 → 卷积滑动 → 池化 → 卷积核调参")

    if not os.path.exists(CSV_PATH):
        st.error(f"未找到 {CSV_PATH}")
        return
    df = safe_read_csv(CSV_PATH)
    if "text" not in df.columns or "label" not in df.columns:
        st.error("CSV 必须含 text 和 label 两列")
        return
    unique = list(pd.unique(df["label"]))
    if len(unique) != 2:
        st.error(f"label 必须是 2 类，当前为 {unique}")
        return

    texts = df["text"].astype(str).tolist()
    y = np.array([0 if str(l) == str(unique[0]) else 1 for l in df["label"]])

    @st.cache_resource(show_spinner=False)
    def train_classifier(texts_tuple, y_tuple):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.neural_network import MLPClassifier
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        X = vec.fit_transform(list(texts_tuple))
        clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=SEED)
        clf.fit(X, np.array(y_tuple))
        return vec, clf

    with st.spinner("训练分类器中..."):
        vec, clf = train_classifier(tuple(texts), tuple(y))
    st.success(f"分类器就绪 | 数据 {len(texts)} 条 | 类别 {unique}")

    sentence = st.text_input("输入一句英文：", "I very like this book")
    if st.button("🔮 预测", key="tab2_predict"):
      if not sentence.strip():
        st.warning("⚠️ 请输入句子，不能为空！")
      else:
        X = vec.transform([sentence])
        probs = clf.predict_proba(X)[0]

        chinglish_idx = 1
        for i, c in enumerate(unique):
            cs = str(c).lower()
            if "chinglish" in cs or "中式" in cs or cs in ("1", "yes", "true"):
                chinglish_idx = i
                break
        native_idx = 1 - chinglish_idx

        st.subheader("预测结果")
        st.write("**中式英语** 概率")
        st.progress(float(probs[chinglish_idx]))
        st.write(f"{probs[chinglish_idx] * 100:.2f}%")

        st.write("**地道英语** 概率")
        st.progress(float(probs[native_idx]))
        st.write(f"{probs[native_idx] * 100:.2f}%")
        chinglish_idx = 1
        for i, c in enumerate(unique):
            cs = str(c).lower()
            if "chinglish" in cs or "中式" in cs or cs in ("1", "yes", "true"):
                chinglish_idx = i
                break
        native_idx = 1 - chinglish_idx
        st.subheader("预测结果")
        st.write("**中式英语** 概率")
        st.progress(float(probs[chinglish_idx]))
        st.write(f"{probs[chinglish_idx] * 100:.2f}%")
        st.write("**地道英语** 概率")
        st.progress(float(probs[native_idx]))
        st.write(f"{probs[native_idx] * 100:.2f}%")

    st.markdown("---")
    st.subheader("🧠 TextCNN 内部计算流程可视化")

    ks_show = st.slider("卷积核大小 kernel_size（改完重新可视化）", 2, 5, 3)
    show_slide = st.checkbox("显示卷积滑动动画（逐位置）", value=False)

    if st.button("🎨 可视化卷积与池化"):
        tokens = tokenize(sentence)
        if len(tokens) < 2:
            st.warning("句子太短，至少 2 个词。")
            return

        # 尝试用 Word2Vec 获取向量，失败则随机
        embed_matrix = None
        if os.path.isdir(W2V_MODEL_DIR):
            cands = sorted([f for f in os.listdir(W2V_MODEL_DIR) if f.endswith(".model")])
            if cands:
                try:
                    from gensim.models import Word2Vec
                    w2v = Word2Vec.load(os.path.join(W2V_MODEL_DIR, cands[0]))
                    embed_dim = w2v.vector_size
                    vectors = []
                    for t in tokens:
                        if t in w2v.wv:
                            vectors.append(w2v.wv[t])
                        else:
                            rng = np.random.RandomState(hash(t) % 2**31)
                            vectors.append(rng.randn(embed_dim))
                    embed_matrix = np.array(vectors)
                    st.caption(f"使用 Word2Vec 向量（维度 {embed_dim}）")
                except Exception:
                    embed_matrix = None

        if embed_matrix is None:
            rng = np.random.RandomState(SEED)
            embed_matrix = rng.randn(len(tokens), 32)
            st.caption("使用随机初始化向量（维度 32）")

        st.write(f"**分词结果：** {tokens}")
        st.write(f"**词向量矩阵形状：** {embed_matrix.shape[0]} × {embed_matrix.shape[1]}")

        results = simulate_cnn(embed_matrix, kernel_sizes=(2, 3, 4, 5), num_filters=4)
        if not results:
            st.warning("句子太短，无法做卷积。")
            return

        fig = draw_cnn_visual(embed_matrix, tokens, results, ks_show)
        st.pyplot(fig)

        if show_slide:
            fig2 = draw_conv_sliding_anim(embed_matrix, tokens, ks_show)
            st.pyplot(fig2)


# ==================== Tab 3: 多模型对照 ====================
def tab_compare():
    st.header("Tab 3 · 多模型对照实验工作台")
    st.caption("玩法：同一句话切换不同模型 → 看预测标签 / 置信度 / 批量对比")

    if not os.path.exists(CSV_PATH):
        st.error(f"未找到 {CSV_PATH}")
        return
    df = safe_read_csv(CSV_PATH)
    if "text" not in df.columns or "label" not in df.columns:
        st.error("CSV 必须含 text 和 label 两列")
        return
    unique = list(pd.unique(df["label"]))
    if len(unique) != 2:
        st.error(f"label 必须是 2 类")
        return

    texts = df["text"].astype(str).tolist()
    y = np.array([0 if str(l) == str(unique[0]) else 1 for l in df["label"]])

    @st.cache_resource(show_spinner=False)
    def train_all(texts_tuple, y_tuple):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.svm import LinearSVC
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.neural_network import MLPClassifier

        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        X = vec.fit_transform(list(texts_tuple))
        y_arr = np.array(y_tuple)

        nb = MultinomialNB().fit(X, y_arr)
        svm = CalibratedClassifierCV(LinearSVC(max_iter=2000), cv=3).fit(X, y_arr)
        mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300,
                            random_state=SEED).fit(X, y_arr)
        return vec, nb, svm, mlp

    with st.spinner("训练三个模型..."):
        vec, nb, svm, mlp = train_all(tuple(texts), tuple(y))

    model_name = st.selectbox("选择模型", ["朴素贝叶斯", "SVM", "MLP（替代TextCNN）"])
    sentence = st.text_input("输入英文句子", "I very like this food")

    if st.button("🔮 预测", key="tab3_predict"):
      if not sentence.strip():
        st.warning("⚠️ 请输入句子，不能为空！")
      else:
        X = vec.transform([sentence])
        if model_name == "朴素贝叶斯":
            probs = nb.predict_proba(X)[0]
        elif model_name == "SVM":
            probs = svm.predict_proba(X)[0]
        else:
            probs = mlp.predict_proba(X)[0]

        pred = int(np.argmax(probs))
        label_name = {0: str(unique[0]), 1: str(unique[1])}
        st.success(f"【{model_name}】预测：**{label_name[pred]}** 置信度：**{probs[pred] * 100:.2f}%**")

        st.write("各类别概率：")
        for i, u in enumerate(unique):
            st.write(f"- {u}: {probs[i] * 100:.2f}%")
            st.progress(float(probs[i]))

        pred = int(np.argmax(probs))
        conf = float(probs[pred])
        label_name = {0: str(unique[0]), 1: str(unique[1])}
        st.success(f"【{model_name}】预测：**{label_name[pred]}** 置信度：**{conf * 100:.2f}%**")
        st.write("各类别概率：")
        for i, u in enumerate(unique):
            st.write(f"- {u}: {probs[i] * 100:.2f}%")
            st.progress(float(probs[i]))

        st.markdown("---")
    st.subheader("📊 三模型批量对比")

    if st.button("对当前句子跑三个模型", key="tab3_batch"):
        tokens = tokenize(sentence)
        if not sentence.strip():
            st.warning("⚠️ 请输入句子，不能为空！")
        elif len([t for t in tokens if t.isalpha()]) == 0:
            st.warning("⚠️ 输入里没有有效英文单词，请重新输入！")
        else:
            X = vec.transform([sentence])
            rows = []
            for name, clf in [("朴素贝叶斯", nb), ("SVM", svm), ("MLP", mlp)]:
                p = clf.predict_proba(X)[0]
                rows.append({
                    "模型": name,
                    "预测": str(unique[int(np.argmax(p))]),
                    "置信度": f"{float(p.max()) * 100:.2f}%",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True)


# ==================== 主入口 ====================
def main():
    st.set_page_config(page_title="Chinglish NLP 演示", layout="wide")
    st.title("Chinglish NLP 演示 · 词向量 / TextCNN / 多模型对照")

    tab1, tab2, tab3 = st.tabs(["Tab1 词向量调参", "Tab2 CNN 文本分类", "Tab3 多模型对照"])
    with tab1:
        tab_word2vec()
    with tab2:
        tab_cnn()
    with tab3:
        tab_compare()


if __name__ == "__main__":
    main()
