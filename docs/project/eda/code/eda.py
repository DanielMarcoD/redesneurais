"""EDA Online Shoppers: execute este arquivo a partir de qualquer diretório.

Imprime tabelas em Markdown e gera oito figuras; não modifica o relatório.
Não treina classificadores. Importar o módulo não executa a análise.
"""

from hashlib import sha256
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.manifold import TSNE
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


SEED = 42
PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[4]
DATA_PATH = REPO_ROOT / "data/online-shoppers/online_shoppers_intention.csv"
FIGURES_DIR = PROJECT_DIR / "figures"
CSV_SHA256 = "b3055ee355f59134d851d32641183cb4a8b45def7124d2f50442a042f358e0d9"
COUNTS = ["Administrative", "Informational", "ProductRelated"]
DURATIONS = [f"{column}_Duration" for column in COUNTS]
NUMERIC_ORIGINAL = [
    "Administrative", "Administrative_Duration", "Informational",
    "Informational_Duration", "ProductRelated", "ProductRelated_Duration",
    "BounceRates", "ExitRates", "PageValues", "SpecialDay",
]
NUMERIC = [column for column in NUMERIC_ORIGINAL if column != "PageValues"]
CATEGORICAL = [
    "Month", "OperatingSystems", "Browser", "Region", "TrafficType",
    "VisitorType", "Weekend",
]
RETAINED = NUMERIC + CATEGORICAL
EXPECTED_COLUMNS = NUMERIC_ORIGINAL + CATEGORICAL + ["Revenue"]
BOX_COMPARISONS = [
    ("ProductRelated_Duration", "Revenue"), ("ExitRates", "Revenue"),
    ("ProductRelated_Duration", "VisitorType"), ("ExitRates", "VisitorType"),
]


def load_data():
    """Lê o CSV versionado e verifica a identidade e o esquema da fonte."""
    if sha256(DATA_PATH.read_bytes()).hexdigest() != CSV_SHA256:
        raise ValueError("CSV diferente da fonte oficial: confira o SHA-256.")
    data = pd.read_csv(DATA_PATH)
    assert list(data.columns) == EXPECTED_COLUMNS
    assert data.shape == (12330, 18)
    assert data[NUMERIC_ORIGINAL].apply(pd.api.types.is_numeric_dtype).all()
    assert pd.api.types.is_bool_dtype(data["Revenue"])
    assert pd.api.types.is_bool_dtype(data["Weekend"])
    assert set(data["Revenue"]) == {False, True}
    return data


def split_data(data):
    """Separa aproximadamente 80/20 sem compartilhar preditoras idênticas."""
    x = data.drop(columns="Revenue")
    y = data["Revenue"]
    groups = pd.util.hash_pandas_object(x[RETAINED], index=False)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    train_index, test_index = next(splitter.split(x, y, groups))
    x_train, x_test = x.iloc[train_index].copy(), x.iloc[test_index].copy()
    y_train, y_test = y.iloc[train_index].copy(), y.iloc[test_index].copy()
    assert set(x_train.index).isdisjoint(x_test.index)
    assert set(groups.iloc[train_index]).isdisjoint(groups.iloc[test_index])
    assert len(x_train) + len(x_test) == len(data)
    for target in (y_train, y_test):
        assert set(target) == {False, True}
        assert abs(target.mean() - y.mean()) < 0.01
    return x_train, x_test, y_train, y_test


def build_preprocessor():
    """Constrói um pré-processador sem ajustar estatísticas ou categorias."""
    return ColumnTransformer([
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), NUMERIC),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), CATEGORICAL),
    ], remainder="drop")


def quality_tables(data):
    """Auditoria estrutural global, sem estimar transformações."""
    missing = pd.DataFrame({
        "Coluna": data.columns, "Tipo no CSV": data.dtypes.astype(str).values,
        "Ausentes": data.isna().sum().values,
        "Ausentes (%)": data.isna().mean().values * 100,
        "Cardinalidade": data.nunique().values,
    })
    checks = {}
    for column in COUNTS:
        checks[f"{column}: negativo ou fracionário"] = int(
            ((data[column] < 0) | (data[column] % 1 != 0)).sum())
    for column in DURATIONS + ["PageValues"]:
        checks[f"{column}: negativo"] = int((data[column] < 0).sum())
    for column in ["BounceRates", "ExitRates", "SpecialDay"]:
        checks[f"{column}: fora de [0,1]"] = int((~data[column].between(0, 1)).sum())
    months = ["Jan", "Feb", "Mar", "Apr", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    checks["Month: rótulo inesperado"] = int((~data["Month"].isin(months)).sum())
    checks["VisitorType: rótulo inesperado"] = int((~data["VisitorType"].isin(
        ["New_Visitor", "Returning_Visitor", "Other"])).sum())
    for column in ["OperatingSystems", "Browser", "Region", "TrafficType"]:
        checks[f"{column}: código não inteiro positivo"] = int(
            ((data[column] <= 0) | (data[column] % 1 != 0)).sum())
    checks["Numéricas: valores não finitos, exceto ausentes"] = int(
        (np.isinf(data[NUMERIC_ORIGINAL].to_numpy())).sum())
    return missing, pd.DataFrame(checks.items(), columns=["Verificação", "Ocorrências"])


def training_tables(x_train, y_train):
    """Calcula as tabelas que orientarão decisões, exclusivamente no treino."""
    numbers = x_train[NUMERIC_ORIGINAL]
    stats = numbers.describe().T.rename(columns={
        "count": "n", "mean": "Média", "std": "Desvio", "min": "Mínimo",
        "25%": "Q1", "50%": "Mediana", "75%": "Q3", "max": "Máximo",
    })
    stats["IQR"] = stats["Q3"] - stats["Q1"]
    stats["Assimetria"] = numbers.skew()
    stats["Zeros (%)"] = numbers.eq(0).mean() * 100
    outliers = (numbers.lt(stats["Q1"] - 1.5 * stats["IQR"]) |
                numbers.gt(stats["Q3"] + 1.5 * stats["IQR"]))
    stats["Fora do IQR"] = outliers.sum()

    categorical_rows = []
    for column in CATEGORICAL:
        for category, count in x_train[column].value_counts().items():
            purchases = int(y_train[x_train[column] == category].sum())
            categorical_rows.append([
                column, str(category), int(count), count / len(x_train) * 100,
                purchases, int(count) - purchases, purchases / count * 100,
                count / len(x_train) < 0.01,
            ])
    categorical = pd.DataFrame(categorical_rows, columns=[
        "Variável", "Categoria", "n", "% treino", "Compras", "Sem compra",
        "Compra (%)", "Rara (<1%)",
    ])
    cardinality = pd.DataFrame([
        [column, x_train[column].nunique(),
         int((categorical.loc[categorical["Variável"] == column, "% treino"] < 1).sum())]
        for column in CATEGORICAL
    ], columns=["Variável", "Cardinalidade no treino", "Categorias raras"])

    correlation = numbers.corr(method="spearman")
    pairs = pd.DataFrame([
        [left, right, correlation.loc[left, right]]
        for left, right in combinations(NUMERIC_ORIGINAL, 2)
    ], columns=["Variável 1", "Variável 2", "Spearman"])
    pairs = pairs.iloc[np.argsort(-pairs["Spearman"].abs().to_numpy(), kind="stable")].reset_index(drop=True)

    grouped_rows = []
    joined = x_train.assign(Revenue=y_train)
    for numeric, categorical_column in BOX_COMPARISONS:
        for group, subset in joined.groupby(categorical_column, observed=True):
            values = subset[numeric]
            q1, q3 = values.quantile([0.25, 0.75])
            grouped_rows.append([
                numeric, categorical_column, str(group), len(values),
                values.median(), q1, q3, q3 - q1,
            ])
    grouped = pd.DataFrame(grouped_rows, columns=[
        "Numérica", "Agrupamento", "Grupo", "n", "Mediana", "Q1", "Q3", "IQR",
    ])
    return stats, outliers, categorical, cardinality, correlation, pairs, grouped


def validate_preprocessor(preprocess, x_train, x_test, train_t):
    """Confere origem das estatísticas, dados novos e invariância no transform."""
    numeric = preprocess.named_transformers_["num"]
    categorical = preprocess.named_transformers_["cat"]
    imputer = numeric.named_steps["impute"]
    scaler = numeric.named_steps["scale"]
    encoder = categorical.named_steps["encode"]
    np.testing.assert_allclose(imputer.statistics_, x_train[NUMERIC].median())
    filled = x_train[NUMERIC].fillna(x_train[NUMERIC].median()).to_numpy()
    np.testing.assert_allclose(scaler.mean_, filled.mean(axis=0))
    np.testing.assert_allclose(scaler.var_, filled.var(axis=0))
    assert scaler.n_samples_seen_ == len(x_train)
    for i, column in enumerate(CATEGORICAL):
        mode = x_train[column].mode().iloc[0]
        assert categorical.named_steps["impute"].statistics_[i] == mode
        np.testing.assert_array_equal(
            encoder.categories_[i], np.sort(x_train[column].fillna(mode).unique()))

    fitted_arrays = [imputer.statistics_, scaler.mean_, scaler.var_, scaler.scale_,
                     categorical.named_steps["impute"].statistics_, *encoder.categories_]
    before = [array.copy() for array in fitted_arrays]
    test_t = preprocess.transform(x_test)
    names = preprocess.get_feature_names_out()
    assert not any("Revenue" in name or "PageValues" in name for name in names)
    assert train_t.shape == (len(x_train), len(names))
    assert test_t.shape == (len(x_test), len(names))
    assert np.isfinite(train_t).all() and np.isfinite(test_t).all()

    # Sondas são cópias do treino; não alteram dados, categorias ou estatísticas.
    missing_probe = x_train.iloc[[0]].copy().astype(object)
    missing_probe.loc[:, RETAINED] = np.nan
    assert np.isfinite(preprocess.transform(missing_probe)).all()
    unknown_probe = x_train.iloc[[0]].copy()
    unknown_probe["VisitorType"] = "__categoria_nova__"
    unknown_t = preprocess.transform(unknown_probe)
    visitor_block = np.array([name.startswith("cat__VisitorType_") for name in names])
    assert visitor_block.any() and (unknown_t[:, visitor_block] == 0).all()
    assert np.isfinite(unknown_t).all()
    changed_excluded = x_test.copy()
    changed_excluded["PageValues"] = -999
    np.testing.assert_array_equal(preprocess.transform(changed_excluded), test_t)
    for array, previous in zip(fitted_arrays, before):
        np.testing.assert_array_equal(array, previous)
    return test_t, names


def print_table(title, table):
    """Markdown sem depender de tabulate; precisão numérica fixa."""
    print(f"\n### {title}\n")
    print("| " + " | ".join(map(str, table.columns)) + " |")
    print("| " + " | ".join("---" for _ in table.columns) + " |")
    for row in table.itertuples(index=False, name=None):
        cells = [f"{cell:.6f}" if isinstance(cell, (float, np.floating)) else str(cell) for cell in row]
        print("| " + " | ".join(cells) + " |")


def save_figure(fig, number):
    import matplotlib.pyplot as plt

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"figura_{number:02d}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_embedding(ax, coordinates, target, title):
    for value, label, color in [(False, "Sem compra", "tab:blue"), (True, "Com compra", "tab:orange")]:
        mask = target == value
        ax.scatter(coordinates[mask, 0], coordinates[mask, 1], s=9, alpha=0.55,
                   color=color, label=label, edgecolors="none", rasterized=True)
    ax.set(title=title, xlabel="Eixo 1", ylabel="Eixo 2")
    ax.legend(markerscale=2)


def plot_descriptive(x_train, y_train, categorical, correlation, pairs):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 5, figsize=(19, 8))
    for ax, column in zip(axes.flat, NUMERIC_ORIGINAL):
        ax.hist(x_train[column], bins=30, color="tab:blue", edgecolor="white")
        ax.set_yscale("log")
        ax.set(title=column, xlabel="Valor original", ylabel="Sessões (escala log)")
        ax.tick_params(axis="x", labelsize=8)
    fig.suptitle("Figura 1 — Distribuições numéricas no treino", fontsize=14)
    save_figure(fig, 1)

    for number, rate in [(2, False), (5, True)]:
        fig, axes = plt.subplots(3, 3, figsize=(20, 14))
        for ax, column in zip(axes.flat, CATEGORICAL):
            subset = categorical[categorical["Variável"] == column]
            heights = subset["Compra (%)"] if rate else subset["n"]
            positions = np.arange(len(subset))
            ax.bar(positions, heights, color="tab:orange" if rate else "tab:blue")
            labels = [f"{row.Categoria}\n(n={row.n})" if rate else row.Categoria
                      for row in subset.itertuples(index=False)]
            ax.set_xticks(positions, labels, rotation=55, ha="right", fontsize=8)
            ax.set(title=column, xlabel="Categoria", ylabel="Compra (%)" if rate else "Sessões")
            if rate:
                ax.set_ylim(0, 105)
                ax.axhline(y_train.mean() * 100, color="black", linestyle="--", label="Taxa geral do treino")
                ax.legend(fontsize=8)
        for ax in list(axes.flat)[len(CATEGORICAL):]:
            ax.set_visible(False)
        title = "Taxa de compra por categoria" if rate else "Frequências categóricas"
        fig.suptitle(f"Figura {number} — {title} no treino", fontsize=15)
        save_figure(fig, number)

    fig, ax = plt.subplots(figsize=(13, 11))
    matrix = ax.imshow(correlation, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(10), NUMERIC_ORIGINAL, rotation=55, ha="right")
    ax.set_yticks(range(10), NUMERIC_ORIGINAL)
    for i in range(10):
        for j in range(10):
            value = correlation.iloc[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center",
                    color="white" if abs(value) > 0.6 else "black", fontsize=9)
    ax.set(title="Figura 3 — Correlação de Spearman no treino", xlabel="Variável", ylabel="Variável")
    fig.colorbar(matrix, ax=ax, label="Spearman (ρ)")
    save_figure(fig, 3)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, row in zip(axes, pairs.head(2).itertuples(index=False, name=None)):
        left, right, coefficient = row
        ax.scatter(x_train[left], x_train[right], s=5, alpha=0.25, label="Sessões do treino")
        ax.set(xlabel=left, ylabel=right, title=f"Spearman = {coefficient:.4f}")
        ax.legend()
    fig.suptitle("Figura 4 — Dois pares numéricos mais associados", fontsize=14)
    save_figure(fig, 4)

    joined = x_train.assign(Revenue=y_train)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, (numeric, category) in zip(axes.flat, BOX_COMPARISONS):
        levels = sorted(joined[category].unique())
        ax.boxplot([joined.loc[joined[category] == level, numeric] for level in levels],
                   tick_labels=[str(level) for level in levels],
                   flierprops={"markersize": 2, "alpha": 0.3})
        ax.set(title=f"{numeric} por {category}", xlabel=category, ylabel=numeric)
        if numeric.endswith("Duration"):
            ax.set_yscale("symlog", linthresh=1)
            ax.set_ylabel(f"{numeric} (escala symlog)")
    fig.suptitle("Figura 6 — Localização e dispersão por grupo no treino", fontsize=14)
    save_figure(fig, 6)


def dimensionality_reduction(train_t, y_train, feature_names):
    """PCA no treino completo; métodos não lineares em uma amostra comum."""
    import matplotlib.pyplot as plt
    from umap import UMAP

    pca = PCA(svd_solver="full").fit(train_t)
    variance = pd.DataFrame({
        "Componente": np.arange(1, len(pca.explained_variance_ratio_) + 1),
        "Variância (%)": pca.explained_variance_ratio_ * 100,
        "Acumulada (%)": np.cumsum(pca.explained_variance_ratio_) * 100,
    })
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    loading_rows = []
    for component in range(2):
        indices = np.argsort(-np.abs(loadings[:, component]))[:5]
        for index in indices:
            loading_rows.append([f"PC{component + 1}", feature_names[index], loadings[index, component]])

    sample_positions, _ = train_test_split(
        np.arange(len(train_t)), train_size=2000, stratify=y_train, random_state=SEED)
    sample = train_t[sample_positions]
    target = y_train.iloc[sample_positions].to_numpy()
    assert len(np.unique(sample_positions)) == 2000
    assert abs(target.mean() - y_train.mean()) < 0.01
    pca_projection = pca.transform(sample)[:, :2]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(variance["Componente"], variance["Acumulada (%)"], marker=".")
    for threshold in (90, 95):
        axes[0].axhline(threshold, linestyle="--", color="gray", label=f"{threshold}%")
    axes[0].set(xlabel="Número de componentes", ylabel="Variância acumulada (%)", title="PCA no treino completo")
    axes[0].legend()
    plot_embedding(axes[1], pca_projection, target, "PCA — amostra de 2.000 sessões")
    axes[1].set_xlabel(f"PC1 ({variance.iloc[0]['Variância (%)']:.2f}%)")
    axes[1].set_ylabel(f"PC2 ({variance.iloc[1]['Variância (%)']:.2f}%)")
    fig.suptitle("Figura 7 — Variância explicada e projeção PCA", fontsize=14)
    save_figure(fig, 7)

    projections = []
    for perplexity in (5, 30):
        projection = TSNE(
            n_components=2, perplexity=perplexity, init="pca", learning_rate="auto",
            max_iter=1000, metric="euclidean", random_state=SEED, n_jobs=1,
        ).fit_transform(sample)
        projections.append((f"t-SNE — perplexity={perplexity}", projection))
    for neighbors in (15, 50):
        # Sem y: o alvo só estratifica a amostra e colore a visualização.
        projection = UMAP(
            n_components=2, n_neighbors=neighbors, min_dist=0.1, n_epochs=500,
            metric="euclidean", random_state=SEED, n_jobs=1,
        ).fit_transform(sample)
        projections.append((f"UMAP — n_neighbors={neighbors}", projection))
    for projection in [pca_projection] + [coordinates for _, coordinates in projections]:
        assert projection.shape == (2000, 2) and np.isfinite(projection).all()
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    for ax, (title, projection) in zip(axes.flat, projections):
        plot_embedding(ax, projection, target, title)
    fig.suptitle("Figura 8 — Mesmas 2.000 sessões em t-SNE e UMAP", fontsize=14)
    save_figure(fig, 8)
    return variance, pd.DataFrame(loading_rows, columns=["Componente", "Feature", "Loading"]), target


def main():
    import matplotlib

    matplotlib.use("Agg")
    data = load_data()
    missing, domains = quality_tables(data)
    assert domains["Ocorrências"].sum() == 0
    x_train, x_test, y_train, y_test = split_data(data)
    stats, outliers, categories, cardinality, correlation, pairs, grouped = training_tables(x_train, y_train)
    preprocess = build_preprocessor()
    train_t = preprocess.fit_transform(x_train)
    test_t, feature_names = validate_preprocessor(preprocess, x_train, x_test, train_t)

    print_table("Qualidade global", missing)
    print_table("Domínios e inconsistências", domains)
    groups = pd.util.hash_pandas_object(data[RETAINED], index=False)
    conflicts = data.groupby(groups)["Revenue"].nunique().gt(1).sum()
    print_table("Repetições e constantes", pd.DataFrame([
        ["Repetições completas além da primeira", int(data.duplicated().sum())],
        ["Repetições das 17 preditoras além da primeira", int(data.drop(columns="Revenue").duplicated().sum())],
        ["Repetições das 16 preditoras retidas além da primeira", int(data[RETAINED].duplicated().sum())],
        ["Grupos com alvos conflitantes (16 preditoras)", int(conflicts)],
        ["Colunas constantes", int(data.nunique().le(1).sum())],
    ], columns=["Indicador", "Valor"]))
    target_rows = []
    for label, target in [("Base completa", data["Revenue"]), ("Treino", y_train), ("Teste", y_test)]:
        for value in (False, True):
            count = int(target.eq(value).sum())
            target_rows.append([label, str(value), count, count / len(target) * 100])
    print_table("Distribuição do alvo", pd.DataFrame(target_rows, columns=["Conjunto", "Revenue", "n", "%"]))
    print_table("Estatísticas numéricas do treino", stats.reset_index(names="Variável"))
    print_table("Cardinalidade no treino", cardinality)
    print_table("Categorias e compras no treino", categories)
    print_table("Correlação de Spearman no treino", correlation.reset_index(names="Variável"))
    print_table("Pares numéricos ordenados", pairs)
    print_table("Comparações agrupadas no treino", grouped)
    print_table("Extremos sinalizados (sem remoção)", pd.DataFrame([
        ["Sessões sinalizadas nas 10 numéricas", int(outliers.any(axis=1).sum())],
        ["Sessões sinalizadas nas 9 numéricas retidas", int(outliers[NUMERIC].any(axis=1).sum())],
        ["Sessões removidas ou limitadas", 0],
    ], columns=["Indicador", "n"]))
    unknown_rows = []
    encoder = preprocess.named_transformers_["cat"].named_steps["encode"]
    for column, known in zip(CATEGORICAL, encoder.categories_):
        unknown = x_test.loc[~x_test[column].isin(known), column]
        unknown_rows.append([column, ", ".join(map(str, sorted(unknown.unique()))) or "Nenhuma", len(unknown)])
    print_table("Categorias novas no teste (somente diagnóstico)", pd.DataFrame(
        unknown_rows, columns=["Variável", "Categorias novas", "Sessões"]))
    print_table("Features finais", pd.DataFrame({"Índice": np.arange(len(feature_names)), "Feature": feature_names}))

    plot_descriptive(x_train, y_train, categories, correlation, pairs)
    variance, loadings, sample_target = dimensionality_reduction(train_t, y_train, feature_names)
    print_table("Variância PCA", variance)
    print_table("Maiores loadings", loadings)
    print_table("Amostra comum das projeções", pd.DataFrame([
        [str(value), int((sample_target == value).sum())] for value in (False, True)
    ], columns=["Revenue", "n"]))
    print_table("Resultados finais", pd.DataFrame([
        [1, "Dataset, tarefa e alvo", "Online Shoppers; classificação; Revenue"],
        [2, "Instâncias e preditoras", f"{len(data)} × 17 (10 numéricas / 7 categóricas); retidas: 16 (9 / 7)"],
        [3, "Maior ausência", f"Todas empatadas: {data.isna().mean().max() * 100:.6f}%"],
        [4, "Colunas descartadas", "PageValues: disponibilidade temporal não esclarecida"],
        [5, "Classe minoritária na base (%)", f"{data['Revenue'].mean() * 100:.6f}"],
        [6, "Treino / teste", f"{len(x_train)} / {len(x_test)}"],
        [7, "Maior correlação numérica (treino)", f"{pairs.iloc[0, 0]} × {pairs.iloc[0, 1]}: {pairs.iloc[0, 2]:.6f}"],
        [8, "Linhas removidas ou limitadas por extremos", "0"],
        [9, "Variância PC1 + PC2 (%)", f"{variance.iloc[1]['Acumulada (%)']:.6f}"],
        [10, "Shape após pipeline", f"Treino: {train_t.shape}; teste: {test_t.shape}"],
    ], columns=["#", "Resultado", "Valor"]))
    print("\nValidação: sem NaN/inf; estatísticas e categorias do treino; grupos disjuntos; sondas OK.")


if __name__ == "__main__":
    # Dependência do scikit-learn: evita variações de BLAS/OpenMP entre execuções.
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=1):
        main()
