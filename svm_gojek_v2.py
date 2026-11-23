import re
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from prettytable import PrettyTable

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc
)

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8

# ==========================================================
# 1. LOAD DATASET & PELABELAN SENTIMEN
# ==========================================================
df_raw = pd.read_csv('GojekAppReviewV4.0.0-V4.9.3_Cleaned.csv')
df_raw = df_raw[['content', 'score']].dropna()
rating_counts = df_raw['score'].astype(int).value_counts().sort_index()

df = df_raw[df_raw['score'] != 3].copy()
df['label'] = df['score'].apply(lambda x: 1 if x >= 4 else -1)
df['sentiment'] = df['label'].apply(lambda x: 'Positif' if x == 1 else 'Negatif')

df_pos = df[df['label'] == 1].sample(n=250, random_state=42)
df_neg = df[df['label'] == -1].sample(n=250, random_state=42)
dataset = pd.concat([df_pos, df_neg]).sample(frac=1, random_state=42).reset_index(drop=True)

# ==========================================================
# 2. NORMALISASI BAHASA TIDAK BAKU & TEXT PREPROCESSING
# ==========================================================
SLANG_DICT = {
    'gak': 'tidak', 'ga': 'tidak', 'gk': 'tidak', 'nggak': 'tidak', 'ngga': 'tidak',
    'tak': 'tidak', 'tdk': 'tidak', 'g': 'tidak', 'kagak': 'tidak',
    'eror': 'error', 'lot': 'lambat', 'lemot': 'lambat', 'lelet': 'lambat',
    'bbrp': 'beberapa', 'yg': 'yang', 'dgn': 'dengan', 'utk': 'untuk',
    'sdh': 'sudah', 'udh': 'sudah', 'udah': 'sudah', 'blm': 'belum',
    'krn': 'karena', 'tp': 'tapi', 'dr': 'dari', 'klo': 'kalau', 'kl': 'kalau',
    'bgt': 'sangat', 'banget': 'sangat', 'bener': 'benar', 'bnyk': 'banyak',
    'baguss': 'bagus', 'siip': 'bagus', 'sip': 'bagus', 'sipp': 'bagus',
    'mantab': 'mantap', 'mantapp': 'mantap', 'mntp': 'mantap', 'kerenn': 'keren',
    'oke': 'baik', 'ok': 'baik', 'good': 'bagus', 'awesome': 'bagus',
    'apk': 'aplikasi', 'app': 'aplikasi', 'skrg': 'sekarang', 'lg': 'lagi',
    'aja': 'saja', 'aj': 'saja', 'doang': 'saja', 'gimana': 'bagaimana',
    'gmn': 'bagaimana', 'knp': 'mengapa', 'kenapa': 'mengapa', 'pdhl': 'padahal',
    'emg': 'memang', 'emang': 'memang', 'sy': 'saya', 'gw': 'saya', 'gue': 'saya',
    'lu': 'kamu', 'lo': 'kamu', 'org': 'orang', 'jd': 'jadi', 'jdi': 'jadi',
    'hrs': 'harus', 'sm': 'sama', 'ama': 'sama', 'bs': 'bisa',
    'susah': 'sulit', 'ribet': 'rumit', 'parah': 'buruk', 'jelek': 'buruk',
    'payah': 'buruk', 'kecewa': 'buruk', 'puas': 'bagus', 'membantu': 'bantu'
}

stopword_remover = StopWordRemoverFactory().create_stop_word_remover()
stemmer = StemmerFactory().create_stemmer()

def normalize_slang(text):
    # Pangkas huruf berulang >= 3 (misal: baguuus -> bagus, siiip -> sip)
    text = re.sub(r'(.)\1{2,}', r'\1', text)
    tokens = text.split()
    normalized = [SLANG_DICT.get(tok, tok) for tok in tokens]
    return ' '.join(normalized)

def preprocess_text(text):
    text = str(text).lower()                           # 1. Case folding
    text = re.sub(r'[^a-z\s]', ' ', text)              # 2. Cleaning simbol & angka
    text = re.sub(r'\s+', ' ', text).strip()           # Normalisasi spasi
    text = normalize_slang(text)                       # 3. Normalisasi kata tidak baku
    text = stopword_remover.remove(text)               # 4. Stopword removal
    text = stemmer.stem(text)                          # 5. Stemming Nazief-Adriani
    return text

print("Sedang menjalankan preprocessing dengan kamus normalisasi slang...")
dataset['clean_text'] = dataset['content'].apply(preprocess_text)
dataset = dataset[dataset['clean_text'] != ''].reset_index(drop=True)

print(f"\nDataset Bersih Valid: N = {len(dataset)} ({dataset['sentiment'].value_counts().to_dict()})")
print("\nContoh Hasil Preprocessing + Normalisasi Slang (5 Sampel Pertama):")
for i in range(5):
    print(f"  [{i+1}] Asli   : {dataset.loc[i, 'content']}")
    print(f"      Bersih : {dataset.loc[i, 'clean_text']} (Label: {dataset.loc[i, 'label']})")

# Konversi ke NumPy array murni agar kompatibel dengan GridSearchCV di pandas 3.x
X_text = np.asarray(dataset['clean_text'].tolist(), dtype=object)
y = np.asarray(dataset['label'].tolist(), dtype=int)

# ==========================================================
# 3. NESTED GRIDSEARCHCV (10-FOLD OUTER, 5-FOLD INNER)
#    + PEMISAHAN METRIK MEAN±SD VS OUT-OF-FOLD (OOF)
# ==========================================================
param_grid = {
    'svm__C': [0.01, 0.1, 1, 10, 100],
    'svm__kernel': ['linear']
}

skf_10 = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

fold_records = []
y_true_oof = np.zeros(len(y), dtype=int)
y_pred_oof = np.zeros(len(y), dtype=int)
y_score_oof = np.zeros(len(y), dtype=float)
best_c_list = []

t0 = time.time()
for fold, (train_idx, test_idx) in enumerate(skf_10.split(X_text, y), 1):
    X_train, X_test = X_text[train_idx], X_text[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    base_pipe = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=1000, ngram_range=(1, 2))),
        ('svm', SVC(random_state=42))
    ])
    
    grid = GridSearchCV(
        estimator=base_pipe,
        param_grid=param_grid,
        cv=5,
        scoring='f1',
        n_jobs=1
    )
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    best_c = grid.best_params_['svm__C']
    best_c_list.append(best_c)
    
    y_pred = best_model.predict(X_test)
    y_score = best_model.decision_function(X_test)
    
    y_true_oof[test_idx] = y_test
    y_pred_oof[test_idx] = y_pred
    y_score_oof[test_idx] = y_score
    
    fold_records.append({
        'Fold': fold,
        'Best_C': best_c,
        'Accuracy': accuracy_score(y_test, y_pred) * 100,
        'Precision': precision_score(y_test, y_pred, pos_label=1, zero_division=0) * 100,
        'Recall': recall_score(y_test, y_pred, pos_label=1, zero_division=0) * 100,
        'F1': f1_score(y_test, y_pred, pos_label=1, zero_division=0) * 100
    })
elapsed_tuned = time.time() - t0

df_folds = pd.DataFrame(fold_records)
t_nested = PrettyTable()
t_nested.title = 'Hasil Nested GridSearchCV Linear SVM (10-Fold Outer, 5-Fold Inner)'
t_nested.field_names = ['Fold', 'Best C', 'Accuracy', 'Precision', 'Recall', 'F1-Score']
for _, r in df_folds.iterrows():
    t_nested.add_row([
        int(r['Fold']), r['Best_C'],
        f"{r['Accuracy']:.2f}%", f"{r['Precision']:.2f}%",
        f"{r['Recall']:.2f}%", f"{r['F1']:.2f}%"
    ])
t_nested.add_row(['-'*4, '-'*6, '-'*8, '-'*9, '-'*6, '-'*8])
t_nested.add_row([
    'Mean', f"Mode:{pd.Series(best_c_list).mode()[0]}",
    f"{df_folds['Accuracy'].mean():.2f}%", f"{df_folds['Precision'].mean():.2f}%",
    f"{df_folds['Recall'].mean():.2f}%", f"{df_folds['F1'].mean():.2f}%"
])
t_nested.add_row([
    'SD', '-',
    f"±{df_folds['Accuracy'].std(ddof=1):.2f}%", f"±{df_folds['Precision'].std(ddof=1):.2f}%",
    f"±{df_folds['Recall'].std(ddof=1):.2f}%", f"±{df_folds['F1'].std(ddof=1):.2f}%"
])

oof_acc = accuracy_score(y_true_oof, y_pred_oof) * 100
oof_prec = precision_score(y_true_oof, y_pred_oof, pos_label=1) * 100
oof_rec = recall_score(y_true_oof, y_pred_oof, pos_label=1) * 100
oof_f1 = f1_score(y_true_oof, y_pred_oof, pos_label=1) * 100
fpr_lin, tpr_lin, _ = roc_curve(y_true_oof, y_score_oof, pos_label=1)
oof_auc = auc(fpr_lin, tpr_lin)
oof_cm = confusion_matrix(y_true_oof, y_pred_oof, labels=[-1, 1])

t_nested.add_row([
    'OOF', '-',
    f"{oof_acc:.2f}%", f"{oof_prec:.2f}%", f"{oof_rec:.2f}%", f"{oof_f1:.2f}%"
])
print("\n", t_nested)
print(f"Waktu Nested GridSearchCV: {elapsed_tuned:.2f} detik | OOF ROC-AUC: {oof_auc:.4f}")
print("Out-of-Fold Confusion Matrix (TN, FP / FN, TP):\n", oof_cm)

optimal_C = float(pd.Series(best_c_list).mode()[0])

# ==========================================================
# 4. PERBANDINGAN KERNEL: TRAIN VS VALIDATION ACCURACY
# ==========================================================
kernel_configs = {
    f'Linear SVM (C={optimal_C})': SVC(kernel='linear', C=optimal_C, random_state=42),
    'Polynomial (d=2, C=1)': SVC(kernel='poly', degree=2, C=1.0, gamma='scale', random_state=42),
    'Polynomial (d=3, C=1)': SVC(kernel='poly', degree=3, C=1.0, gamma='scale', random_state=42)
}

t_kern = PrettyTable()
t_kern.title = 'Perbandingan Kernel SVM: Train vs Validation (10-Fold CV)'
t_kern.field_names = ['Model Kernel', 'Train Acc (Mean±SD)', 'Val Acc (Mean±SD)', 'Gap (Train-Val)', 'OOF F1', 'OOF AUC']

kernel_plot_data = {}
for k_name, k_clf in kernel_configs.items():
    tr_accs, va_accs = [], []
    k_oof_pred = np.zeros(len(y), dtype=int)
    k_oof_score = np.zeros(len(y), dtype=float)
    
    for tr_idx, va_idx in skf_10.split(X_text, y):
        pipe = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=1000, ngram_range=(1, 2))),
            ('svm', k_clf)
        ])
        pipe.fit(X_text[tr_idx], y[tr_idx])
        tr_accs.append(accuracy_score(y[tr_idx], pipe.predict(X_text[tr_idx])) * 100)
        va_pred = pipe.predict(X_text[va_idx])
        va_accs.append(accuracy_score(y[va_idx], va_pred) * 100)
        k_oof_pred[va_idx] = va_pred
        k_oof_score[va_idx] = pipe.decision_function(X_text[va_idx])
        
    tr_m, tr_s = np.mean(tr_accs), np.std(tr_accs, ddof=1)
    va_m, va_s = np.mean(va_accs), np.std(va_accs, ddof=1)
    gap = tr_m - va_m
    k_f1 = f1_score(y, k_oof_pred, pos_label=1) * 100
    fpr_k, tpr_k, _ = roc_curve(y, k_oof_score, pos_label=1)
    k_auc = auc(fpr_k, tpr_k)
    
    kernel_plot_data[k_name] = {'tr_m': tr_m, 'tr_s': tr_s, 'va_m': va_m, 'va_s': va_s, 'fpr': fpr_k, 'tpr': tpr_k, 'auc': k_auc}
    t_kern.add_row([
        k_name, f"{tr_m:.2f}±{tr_s:.2f}%", f"{va_m:.2f}±{va_s:.2f}%",
        f"{gap:.2f}%", f"{k_f1:.2f}%", f"{k_auc:.4f}"
    ])
print("\n", t_kern)

# ==========================================================
# 5. EKSPERIMEN ABLASI N-GRAM (UNIGRAM VS BIGRAM VS GABUNGAN)
# ==========================================================
ngram_configs = {
    'Unigram (1,1)': (1, 1),
    'Bigram (2,2)': (2, 2),
    'Unigram+Bigram (1,2)': (1, 2)
}

t_ng = PrettyTable()
t_ng.title = 'Perbandingan Fitur N-Gram pada Linear SVM (10-Fold CV)'
t_ng.field_names = ['Representasi N-Gram', 'Val Acc (Mean±SD)', 'Val F1 (Mean±SD)', 'OOF Acc', 'OOF F1']

ngram_plot_data = {}
for ng_name, ng_range in ngram_configs.items():
    accs, f1s = [], []
    ng_oof = np.zeros(len(y), dtype=int)
    for tr_idx, va_idx in skf_10.split(X_text, y):
        pipe = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=1000, ngram_range=ng_range)),
            ('svm', SVC(kernel='linear', C=optimal_C, random_state=42))
        ])
        pipe.fit(X_text[tr_idx], y[tr_idx])
        pred = pipe.predict(X_text[va_idx])
        ng_oof[va_idx] = pred
        accs.append(accuracy_score(y[va_idx], pred) * 100)
        f1s.append(f1_score(y[va_idx], pred, pos_label=1, zero_division=0) * 100)
        
    ngram_plot_data[ng_name] = {'acc_m': np.mean(accs), 'acc_s': np.std(accs, ddof=1), 'f1_m': np.mean(f1s), 'f1_s': np.std(f1s, ddof=1)}
    t_ng.add_row([
        ng_name,
        f"{np.mean(accs):.2f}±{np.std(accs, ddof=1):.2f}%",
        f"{np.mean(f1s):.2f}±{np.std(f1s, ddof=1):.2f}%",
        f"{accuracy_score(y, ng_oof)*100:.2f}%",
        f"{f1_score(y, ng_oof, pos_label=1)*100:.2f}%"
    ])
print("\n", t_ng)

# ==========================================================
# 6. UJI STABILITAS VALIDASI: 5-FOLD VS 10-FOLD STRATIFIED CV
# ==========================================================
t_cv = PrettyTable()
t_cv.title = 'Perbandingan Stabilitas Validasi: 5-Fold vs 10-Fold CV'
t_cv.field_names = ['Skema CV', 'Accuracy (Mean±SD)', 'Precision (Mean±SD)', 'Recall (Mean±SD)', 'F1-Score (Mean±SD)', 'OOF F1']

cv_plot_data = {}
for k_val in [5, 10]:
    skf_k = StratifiedKFold(n_splits=k_val, shuffle=True, random_state=42)
    a_l, p_l, r_l, f_l = [], [], [], []
    oof_k = np.zeros(len(y), dtype=int)
    for tr_idx, va_idx in skf_k.split(X_text, y):
        pipe = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=1000, ngram_range=(1, 2))),
            ('svm', SVC(kernel='linear', C=optimal_C, random_state=42))
        ])
        pipe.fit(X_text[tr_idx], y[tr_idx])
        pred = pipe.predict(X_text[va_idx])
        oof_k[va_idx] = pred
        a_l.append(accuracy_score(y[va_idx], pred) * 100)
        p_l.append(precision_score(y[va_idx], pred, pos_label=1) * 100)
        r_l.append(recall_score(y[va_idx], pred, pos_label=1) * 100)
        f_l.append(f1_score(y[va_idx], pred, pos_label=1) * 100)
        
    cv_plot_data[f"{k_val}-Fold CV"] = {'acc_m': np.mean(a_l), 'acc_s': np.std(a_l, ddof=1), 'f1_m': np.mean(f_l), 'f1_s': np.std(f_l, ddof=1)}
    t_cv.add_row([
        f"{k_val}-Fold Stratified",
        f"{np.mean(a_l):.2f}±{np.std(a_l, ddof=1):.2f}%",
        f"{np.mean(p_l):.2f}±{np.std(p_l, ddof=1):.2f}%",
        f"{np.mean(r_l):.2f}±{np.std(r_l, ddof=1):.2f}%",
        f"{np.mean(f_l):.2f}±{np.std(f_l, ddof=1):.2f}%",
        f"{f1_score(y, oof_k, pos_label=1)*100:.2f}%"
    ])
print("\n", t_cv)

# ==========================================================
# 7. EKSTRAKSI SAMPEL ERROR AKTUAL (FALSE POSITIVE & FALSE NEGATIVE)
# ==========================================================
dataset['y_pred_oof'] = y_pred_oof
dataset['decision_score'] = y_score_oof

fp_samples = dataset[(dataset['label'] == -1) & (dataset['y_pred_oof'] == 1)].head(5)
fn_samples = dataset[(dataset['label'] == 1) & (dataset['y_pred_oof'] == -1)].head(5)

print("\n=== SAMPEL AKTUAL FALSE POSITIVE (Aktual Negatif -1, Prediksi Positif +1) ===")
for idx, row in fp_samples.iterrows():
    print(f"  * Asli   : {row['content']}")
    print(f"    Bersih : {row['clean_text']} | f(x) = {row['decision_score']:+.4f}")

print("\n=== SAMPEL AKTUAL FALSE NEGATIVE (Aktual Positif +1, Prediksi Negatif -1) ===")
for idx, row in fn_samples.iterrows():
    print(f"  * Asli   : {row['content']}")
    print(f"    Bersih : {row['clean_text']} | f(x) = {row['decision_score']:+.4f}")

# ==========================================================
# 8. GENERATE 5 GRAFIK PNG RESOLUSI TINGGI (300 DPI)
# ==========================================================
# FIG 1: Distribusi Rating & Sampel
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
colors_bar = ['#d9534f', '#f0ad4e', '#95a5a6', '#5bc0de', '#2ecc71']
bars = axes[0].bar([f"{s} Bintang" for s in rating_counts.index], rating_counts.values / len(df_raw) * 100, color=colors_bar, edgecolor='black', linewidth=0.6)
axes[0].set_title(f'Distribusi Rating Dataset Mentah (N = {len(df_raw):,})', fontsize=11, fontweight='bold')
axes[0].set_ylabel('Persentase (%)', fontsize=10)
axes[0].set_ylim(0, 76)
axes[0].grid(axis='y', linestyle='--', alpha=0.5)
for bar in bars:
    h = bar.get_height()
    axes[0].text(bar.get_x() + bar.get_width()/2, h + 1, f"{h:.1f}%", ha='center', va='bottom', fontsize=9, fontweight='bold')

sent_counts = dataset['sentiment'].value_counts()
axes[1].pie(
    [sent_counts['Negatif'], sent_counts['Positif']],
    labels=[f"Negatif (-1)\n({sent_counts['Negatif']} ulasan)", f"Positif (+1)\n({sent_counts['Positif']} ulasan)"],
    autopct='%1.2f%%', startangle=90, colors=['#e74c3c', '#27ae60'],
    wedgeprops=dict(width=0.45, edgecolor='white', linewidth=2), textprops={'fontsize': 10, 'fontweight': 'bold'}
)
axes[1].set_title(f'Proporsi Sampel Dataset Bersih (N = {len(dataset)})', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('fig1_distribusi_data.png', dpi=300, bbox_inches='tight')
plt.close()

# FIG 2: Train vs Validation Accuracy Antar-Kernel
fig, ax = plt.subplots(figsize=(9, 5))
k_labels = list(kernel_plot_data.keys())
x_k = np.arange(len(k_labels))
w_k = 0.34
tr_means = [kernel_plot_data[k]['tr_m'] for k in k_labels]
tr_stds = [kernel_plot_data[k]['tr_s'] for k in k_labels]
va_means = [kernel_plot_data[k]['va_m'] for k in k_labels]
va_stds = [kernel_plot_data[k]['va_s'] for k in k_labels]

b1 = ax.bar(x_k - w_k/2, tr_means, w_k, yerr=tr_stds, capsize=4, label='Training Accuracy', color='#2980b9', edgecolor='black', linewidth=0.6)
b2 = ax.bar(x_k + w_k/2, va_means, w_k, yerr=va_stds, capsize=4, label='Validation Accuracy (10-Fold)', color='#27ae60', edgecolor='black', linewidth=0.6)
for r in b1:
    h = r.get_height()
    ax.text(r.get_x() + r.get_width()/2, h + 1.5, f"{h:.1f}%", ha='center', va='bottom', fontsize=8.5, fontweight='bold')
for r in b2:
    h = r.get_height()
    ax.text(r.get_x() + r.get_width()/2, h + 1.5, f"{h:.1f}%", ha='center', va='bottom', fontsize=8.5, fontweight='bold')
ax.set_title('Perbandingan Akurasi Training vs. Validation Antar-Kernel SVM', fontsize=11.5, fontweight='bold')
ax.set_ylabel('Accuracy (%)', fontsize=10)
ax.set_xticks(x_k)
ax.set_xticklabels(k_labels, fontsize=9.5, fontweight='bold')
ax.set_ylim(0, 112)
ax.legend(loc='lower right')
ax.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('fig2_train_vs_val_kernel.png', dpi=300, bbox_inches='tight')
plt.close()

# FIG 3: Ablasi N-Gram & Stabilitas 5-Fold vs 10-Fold
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
ng_labels = list(ngram_plot_data.keys())
x_ng = np.arange(len(ng_labels))
b_ng1 = axes[0].bar(x_ng - 0.17, [ngram_plot_data[n]['acc_m'] for n in ng_labels], 0.34, yerr=[ngram_plot_data[n]['acc_s'] for n in ng_labels], capsize=4, label='Accuracy', color='#16a085', edgecolor='black', linewidth=0.6)
b_ng2 = axes[0].bar(x_ng + 0.17, [ngram_plot_data[n]['f1_m'] for n in ng_labels], 0.34, yerr=[ngram_plot_data[n]['f1_s'] for n in ng_labels], capsize=4, label='F1-Score', color='#f39c12', edgecolor='black', linewidth=0.6)
for r in list(b_ng1) + list(b_ng2):
    h = r.get_height()
    axes[0].text(r.get_x() + r.get_width()/2, h + 1.8, f"{h:.1f}%", ha='center', va='bottom', fontsize=8, fontweight='bold')
axes[0].set_title('Perbandingan Representasi Fitur N-Gram', fontsize=11, fontweight='bold')
axes[0].set_xticks(x_ng)
axes[0].set_xticklabels(ng_labels, fontsize=9, fontweight='bold')
axes[0].set_ylim(0, 108)
axes[0].legend(loc='lower right')
axes[0].grid(axis='y', linestyle='--', alpha=0.5)

cv_labels = list(cv_plot_data.keys())
x_cv = np.arange(len(cv_labels))
b_cv1 = axes[1].bar(x_cv - 0.17, [cv_plot_data[c]['acc_m'] for c in cv_labels], 0.34, yerr=[cv_plot_data[c]['acc_s'] for c in cv_labels], capsize=4, label='Accuracy', color='#2e86c1', edgecolor='black', linewidth=0.6)
b_cv2 = axes[1].bar(x_cv + 0.17, [cv_plot_data[c]['f1_m'] for c in cv_labels], 0.34, yerr=[cv_plot_data[c]['f1_s'] for c in cv_labels], capsize=4, label='F1-Score', color='#8e44ad', edgecolor='black', linewidth=0.6)
for r in list(b_cv1) + list(b_cv2):
    h = r.get_height()
    axes[1].text(r.get_x() + r.get_width()/2, h + 1.8, f"{h:.1f}%", ha='center', va='bottom', fontsize=8.5, fontweight='bold')
axes[1].set_title('Stabilitas Performa: 5-Fold vs. 10-Fold Stratified CV', fontsize=11, fontweight='bold')
axes[1].set_xticks(x_cv)
axes[1].set_xticklabels(cv_labels, fontsize=9.5, fontweight='bold')
axes[1].set_ylim(0, 108)
axes[1].legend(loc='lower right')
axes[1].grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('fig3_ngram_dan_cv.png', dpi=300, bbox_inches='tight')
plt.close()

# FIG 4: Out-of-Fold Confusion Matrix & ROC-AUC Curve
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
g_names = ['True Negative (TN)', 'False Positive (FP)', 'False Negative (FN)', 'True Positive (TP)']
g_counts = [f"{v}" for v in oof_cm.flatten()]
g_percs = [f"({v/np.sum(oof_cm)*100:.2f}%)" for v in oof_cm.flatten()]
annot = np.asarray([f"{n}\n{c}\n{p}" for n, c, p in zip(g_names, g_counts, g_percs)]).reshape(2, 2)
sns.heatmap(oof_cm, annot=annot, fmt='', cmap='Blues', xticklabels=['Prediksi Negatif (-1)', 'Prediksi Positif (+1)'], yticklabels=['Aktual Negatif (-1)', 'Aktual Positif (+1)'], annot_kws={'size': 9.5, 'weight': 'bold'}, ax=axes[0])
axes[0].set_title('Out-of-Fold Confusion Matrix (Tuned Linear SVM)', fontsize=11, fontweight='bold')

roc_colors = ['#1e8449', '#d4ac0d', '#c0392b']
for (k_name, m), col in zip(kernel_plot_data.items(), roc_colors):
    axes[1].plot(m['fpr'], m['tpr'], color=col, lw=2.2, label=f"{k_name} (AUC = {m['auc']:.4f})")
axes[1].plot([0, 1], [0, 1], color='gray', lw=1.2, linestyle='--', label='Random (AUC = 0.5000)')
axes[1].set_title('Kurva ROC-AUC Out-of-Fold Antar-Kernel SVM', fontsize=11, fontweight='bold')
axes[1].set_xlabel('False Positive Rate (1 - Specificity)', fontsize=9.5)
axes[1].set_ylabel('True Positive Rate (Recall)', fontsize=9.5)
axes[1].legend(loc='lower right', fontsize=8.5)
axes[1].grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('fig4_confusion_matrix_roc.png', dpi=300, bbox_inches='tight')
plt.close()

# FIG 5: Top 10 Fitur Dominan
final_pipe = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=1000, ngram_range=(1, 2))),
    ('svm', SVC(kernel='linear', C=optimal_C, random_state=42))
])
final_pipe.fit(X_text, y)
feat_names = final_pipe.named_steps['tfidf'].get_feature_names_out()
weights = final_pipe.named_steps['svm'].coef_.toarray().ravel()

top_pos_idx = np.argsort(weights)[-10:]
top_neg_idx = np.argsort(weights)[:10][::-1]

print("\nTop 10 Fitur Positif (+1):", [(feat_names[i], round(float(weights[i]), 4)) for i in top_pos_idx[::-1]])
print("Top 10 Fitur Negatif (-1):", [(feat_names[i], round(float(weights[i]), 4)) for i in top_neg_idx[::-1]])

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
axes[0].barh(feat_names[top_neg_idx], np.abs(weights[top_neg_idx]), color='#e74c3c', edgecolor='black', linewidth=0.6)
axes[0].set_title('Top 10 Fitur Sentimen Negatif (|Bobot w|)', fontsize=11, fontweight='bold')
axes[0].set_xlabel('Magnitudo Bobot SVM (|w|)', fontsize=10)
axes[0].grid(axis='x', linestyle='--', alpha=0.5)

axes[1].barh(feat_names[top_pos_idx], weights[top_pos_idx], color='#27ae60', edgecolor='black', linewidth=0.6)
axes[1].set_title('Top 10 Fitur Sentimen Positif (Bobot w)', fontsize=11, fontweight='bold')
axes[1].set_xlabel('Bobot SVM (w)', fontsize=10)
axes[1].grid(axis='x', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('fig5_top_fitur.png', dpi=300, bbox_inches='tight')
plt.close()

# Uji Inferensi Ulasan Baru
ulasan_baru = [
    "aplikasi sangat membantu driver ramah dan cepat sampai",
    "sering error pas mau bayar gopay saldo kepotong tapi gagal",
    "mantap fitur lengkap promo banyak"
]
print("\nUji Prediksi Ulasan Baru:")
for teks in ulasan_baru:
    bersih = preprocess_text(teks)
    pred = final_pipe.predict([bersih])[0]
    skor = final_pipe.decision_function([bersih])[0]
    print(f"  * '{teks}' -> Bersih: '{bersih}' -> {'Positif (+1)' if pred == 1 else 'Negatif (-1)'} (f(x) = {skor:+.4f})")