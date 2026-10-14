"""MLP NumPy: execute este arquivo para conferir as contas e gerar as 6 figuras.

Não lê dados externos. Requer apenas as dependências do requirements.txt.
Pesos no código: (entradas, saídas); no Exercise 1, a notação é transposta.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Também funciona sem interface gráfica.
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_blobs, make_classification
from sklearn.linear_model import LogisticRegression  # Somente os baselines.
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


def initialize(sizes, rng):
    """Xavier uniforme: W[l] tem shape (sizes[l], sizes[l+1]); biases zero."""
    parameters = []
    for n_in, n_out in zip(sizes[:-1], sizes[1:]):
        limit = np.sqrt(6 / (n_in + n_out))
        weights = rng.uniform(-limit, limit, size=(n_in, n_out))
        parameters.append((weights, np.zeros(n_out)))
    return parameters


def output_activation(logits, task):
    if task == "binary":
        # sigmoid(z) = exp(-log(1+exp(-z))), sem overflow em exp.
        return np.exp(-np.logaddexp(0, -logits))
    if task == "multiclass":
        shifted = logits - logits.max(axis=1, keepdims=True)
        exponentials = np.exp(shifted)
        return exponentials / exponentials.sum(axis=1, keepdims=True)
    raise ValueError("task deve ser 'binary' ou 'multiclass'")


def targets(y, output_size, task):
    """Coluna binária ou matriz one-hot; não usa codificadores do sklearn."""
    return y[:, None] if task == "binary" else np.eye(output_size)[y]


def forward(X, parameters, task):
    """Guarda ativações para aplicar a regra da cadeia depois."""
    activations = [X]
    for layer, (weights, bias) in enumerate(parameters):
        logits = activations[-1] @ weights + bias
        last = layer == len(parameters) - 1
        activations.append(output_activation(logits, task) if last
                           else np.tanh(logits))
    return activations, logits


def loss(logits, y, task):
    """Cross-entropy média a partir dos logits, sem log(0) ou clipping."""
    if task == "binary":
        return float(np.mean(np.logaddexp(0, logits) - y[:, None] * logits))
    shifted = logits - logits.max(axis=1, keepdims=True)
    log_normalizer = np.log(np.exp(shifted).sum(axis=1))
    return float(np.mean(log_normalizer - shifted[np.arange(len(y)), y]))


def backward(parameters, activations, y, task):
    """Mesmo backward em todas as arquiteturas: delta de saída = (p-Y)/N."""
    expected = targets(y, activations[-1].shape[1], task)
    delta = (activations[-1] - expected) / len(y)
    gradients = [None] * len(parameters)
    for layer in reversed(range(len(parameters))):
        weights, bias = parameters[layer]
        d_weights = activations[layer].T @ delta
        d_bias = delta.sum(axis=0)
        assert d_weights.shape == weights.shape and d_bias.shape == bias.shape
        gradients[layer] = (d_weights, d_bias)
        if layer > 0:
            delta = (delta @ weights.T) * (1 - activations[layer] ** 2)
    return gradients


def update(parameters, gradients, learning_rate):
    # Todos os gradientes já foram calculados com os pesos anteriores.
    for (weights, bias), (d_weights, d_bias) in zip(parameters, gradients):
        weights -= learning_rate * d_weights
        bias -= learning_rate * d_bias


def train(X, y, sizes, task, rng, learning_rate=0.1, epochs=1500):
    parameters = initialize(sizes, rng)
    history = []
    # Época 0 = antes de treinar; época 1500 = após 1500 atualizações.
    for epoch in range(epochs + 1):
        activations, logits = forward(X, parameters, task)
        history.append(loss(logits, y, task))
        if epoch < epochs:
            gradients = backward(parameters, activations, y, task)
            update(parameters, gradients, learning_rate)
    assert np.isfinite(history).all()
    assert all(np.isfinite(array).all() for pair in parameters for array in pair)
    return parameters, np.array(history)


def predict(X, parameters, task):
    probabilities = forward(X, parameters, task)[0][-1]
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    if task == "binary":
        return (probabilities[:, 0] >= 0.5).astype(int)
    assert np.allclose(probabilities.sum(axis=1), 1)
    return probabilities.argmax(axis=1)


def parameter_count(parameters):
    return sum(array.size for pair in parameters for array in pair)


def check_gradients():
    """Diferenças centrais em TODOS os pesos/biases das três arquiteturas.

    Dados pequenos e gerador próprio: esta auditoria não consome o rng principal.
    Não participa do treinamento; apenas confere o backward analítico.
    """
    audit_rng = np.random.default_rng(2026)
    errors = {}
    for sizes, task in [([2, 16, 1], "binary"),
                        ([4, 16, 3], "multiclass"),
                        ([4, 16, 16, 3], "multiclass")]:
        X = audit_rng.normal(size=(5, sizes[0]))
        y = np.arange(5) % (2 if task == "binary" else sizes[-1])
        parameters = initialize(sizes, audit_rng)
        activations, _ = forward(X, parameters, task)
        assert [a.shape for a in activations] == [(5, n) for n in sizes]
        gradients = backward(parameters, activations, y, task)
        max_error = 0.0
        for pair, gradient_pair in zip(parameters, gradients):
            for array, analytical in zip(pair, gradient_pair):
                for index in np.ndindex(array.shape):
                    original = array[index]
                    array[index] = original + 1e-5
                    plus = loss(forward(X, parameters, task)[1], y, task)
                    array[index] = original - 1e-5
                    minus = loss(forward(X, parameters, task)[1], y, task)
                    array[index] = original
                    numerical = (plus - minus) / 2e-5
                    max_error = max(max_error, abs(numerical - analytical[index]))
        assert max_error < 1e-7, (sizes, max_error)
        errors[str(sizes)] = max_error
    # Probabilidades e losses corretas mesmo em logits extremos, sem overflow.
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        for task, logits, y in [
            ("binary", np.array([[-1000.], [1000.]]), np.array([1, 0])),
            ("multiclass", np.array([[1000., 0., -1000.]]), np.array([2])),
        ]:
            probabilities = output_activation(logits, task)
            assert np.isfinite(probabilities).all()
            assert np.isfinite(loss(logits, y, task))
            assert np.isclose(loss(logits, y, task), 1000 if task == "binary" else 2000)
    return errors


def check_manual():
    """Conferência NumPy do Exercise 1; a derivação está escrita no relatório."""
    x, y = np.array([0.5, -0.2]), 1.0
    W1 = np.array([[0.3, -0.1], [0.2, 0.4]])
    b1, W2, b2 = np.array([0.1, -0.2]), np.array([0.5, -0.3]), 0.2
    z = W1 @ x + b1
    h = np.tanh(z)
    u = W2 @ h + b2
    prediction = np.tanh(u)
    before = (y - prediction) ** 2
    d_prediction = 2 * (prediction - y)
    d_u = d_prediction * (1 - prediction ** 2)
    d_W2, d_b2 = d_u * h, d_u
    d_h = d_u * W2
    d_z = d_h * (1 - h ** 2)
    d_W1, d_b1 = np.outer(d_z, x), d_z
    W1_new, b1_new = W1 - 0.3 * d_W1, b1 - 0.3 * d_b1
    W2_new, b2_new = W2 - 0.3 * d_W2, b2 - 0.3 * d_b2
    z_new = W1_new @ x + b1_new
    h_new = np.tanh(z_new)
    u_new = W2_new @ h_new + b2_new
    prediction_new = np.tanh(u_new)
    after = (y - prediction_new) ** 2
    assert after < before
    return dict(z=z, h=h, u=u, prediction=prediction, loss=before,
                d_prediction=d_prediction, d_tanh_output=1 - prediction ** 2,
                d_u=d_u, d_W2=d_W2, d_b2=d_b2, d_h=d_h,
                d_tanh_hidden=1 - h ** 2, d_z=d_z, d_W1=d_W1, d_b1=d_b1,
                W1_new=W1_new, b1_new=b1_new, W2_new=W2_new, b2_new=b2_new,
                z_new=z_new, h_new=h_new, u_new=u_new,
                prediction_new=prediction_new, loss_new=after)


def save_figure(fig, directory, number):
    fig.tight_layout()
    fig.savefig(directory / f"fig{number}.png", dpi=160)
    plt.close(fig)


def loss_figure(directory, number, history, title, label):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.arange(len(history)), history, label=label)
    ax.set(title=title, xlabel="Época (atualizações completas)",
           ylabel="Cross-entropy média no treino")
    ax.legend()
    save_figure(fig, directory, number)


def main():
    directory = Path(__file__).resolve().parents[1] / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    gradient_errors = check_gradients()
    manual = check_manual()
    rng = np.random.default_rng(42)
    learning_rate, epochs = 0.1, 1500
    colors = ["#2456a6", "#d46016"]

    # Exercise 2: argumentos e split exatamente como no enunciado.
    X, clusters = make_blobs(
        n_samples=[500, 250, 250],
        centers=np.array([[0., 0.], [3., 3.], [-3., -3.]]),
        cluster_std=1.2, random_state=42)
    y = (clusters > 0).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    fig, ax = plt.subplots(figsize=(7, 5))
    for label, color in enumerate(colors):
        ax.scatter(*X[y == label].T, s=13, alpha=0.65, color=color,
                   label=f"Classe {label}")
    ax.set(title="Figura 1 — Dados binários (1.000 amostras)",
           xlabel="Feature 1", ylabel="Feature 2")
    ax.legend()
    save_figure(fig, directory, 1)

    baseline_binary = LogisticRegression(max_iter=1000).fit(X_train, y_train)
    binary_parameters, binary_loss = train(
        X_train, y_train, [2, 16, 1], "binary", rng, learning_rate, epochs)
    binary_accuracy = np.mean(predict(X_test, binary_parameters, "binary") == y_test)
    loss_figure(directory, 2, binary_loss, "Figura 2 — Treinamento binário",
                "[2, 16, 1] — classes 0 e 1")

    # A grade serve somente à visualização, sem ajustar parâmetros com o teste.
    grid_x, grid_y = np.meshgrid(
        np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 300),
        np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 300))
    grid = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    grid_prob = forward(grid, binary_parameters, "binary")[0][-1][:, 0]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.contourf(grid_x, grid_y, (grid_prob >= 0.5).reshape(grid_x.shape),
                levels=[-0.5, 0.5, 1.5], cmap=ListedColormap(colors), alpha=0.18)
    ax.contour(grid_x, grid_y, grid_prob.reshape(grid_x.shape),
               levels=[0.5], colors="black", linewidths=1)
    for label, color in enumerate(colors):
        ax.scatter(*X_test[y_test == label].T, s=25, color=color,
                   edgecolors="white", linewidths=0.4, label=f"Classe {label}")
    ax.set(title="Figura 3 — Fronteira da MLP e 200 pontos de teste",
           xlabel="Feature 1", ylabel="Feature 2")
    ax.legend(title="Pontos e regiões")
    save_figure(fig, directory, 3)

    # Exercises 3 e 4: mesmos dados brutos e mesmo split para TODAS as execuções.
    X_multi, y_multi = make_classification(
        n_samples=1500, n_features=4, n_informative=4,
        n_redundant=0, n_repeated=0, n_classes=3,
        n_clusters_per_class=2, class_sep=1.0, random_state=42)
    Xm_train, Xm_test, ym_train, ym_test = train_test_split(
        X_multi, y_multi, test_size=0.2, random_state=42, stratify=y_multi)
    baseline_multi = LogisticRegression(max_iter=1000).fit(Xm_train, ym_train)
    runs, curves, models = [], {}, {}
    for seed in (42, 43, 44):
        for sizes in ([4, 16, 3], [4, 16, 16, 3]):
            # Exceção explícita à instância principal: cada par de arquiteturas
            # começa em um gerador novo com a MESMA seed. A execução rasa 42
            # é também o resultado principal do Exercise 3, sem retreinamento.
            pair_rng = np.random.default_rng(seed)
            parameters, history = train(
                Xm_train, ym_train, sizes, "multiclass", pair_rng,
                learning_rate, epochs)
            predictions = predict(Xm_test, parameters, "multiclass")
            runs.append(dict(seed=seed, sizes=sizes,
                             parameters=parameter_count(parameters),
                             initial_loss=history[0], final_loss=history[-1],
                             accuracy=float(np.mean(predictions == ym_test))))
            if seed == 42:
                curves[len(sizes)] = history
                models[len(sizes)] = parameters

    loss_figure(directory, 4, curves[3], "Figura 4 — Treinamento multiclasse",
                "[4, 16, 3] — classes 0, 1 e 2; seed 42")
    predictions = predict(Xm_test, models[3], "multiclass")
    matrix = confusion_matrix(ym_test, predictions, labels=[0, 1, 2])
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix, cmap="Blues", vmin=0)
    for i, j in np.ndindex(matrix.shape):
        ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                color="white" if matrix[i, j] > matrix.max() / 2 else "black")
    ax.set(title="Figura 5 — Matriz de confusão (seed 42)",
           xlabel="Classe predita", ylabel="Classe verdadeira",
           xticks=[0, 1, 2], yticks=[0, 1, 2])
    fig.colorbar(image, ax=ax, label="Número de amostras")
    save_figure(fig, directory, 5)

    fig, ax = plt.subplots(figsize=(7, 4))
    for depth, history in curves.items():
        ax.plot(np.arange(len(history)), history,
                label="[4, 16, 3] — rasa" if depth == 3 else "[4, 16, 16, 3] — profunda")
    ax.set(title="Figura 6 — Profundidade (seed 42; classes 0, 1 e 2)",
           xlabel="Época (atualizações completas)",
           ylabel="Cross-entropy média no treino")
    ax.legend()
    save_figure(fig, directory, 6)

    means = []
    for sizes in ([4, 16, 3], [4, 16, 16, 3]):
        selected = [r for r in runs if r["sizes"] == sizes]
        means.append(dict(sizes=sizes,
                          final_loss=float(np.mean([r["final_loss"] for r in selected])),
                          accuracy=float(np.mean([r["accuracy"] for r in selected]))))
    pair_counts = {f"{i}-{j}": int(matrix[i, j] + matrix[j, i])
                   for i in range(3) for j in range(i + 1, 3)}
    output = dict(
        gradient_max_absolute_errors=gradient_errors, manual=manual,
        learning_rate=learning_rate, epochs=epochs,
        binary=dict(train_counts=np.bincount(y_train), test_counts=np.bincount(y_test),
                    baseline_accuracy=float(np.mean(baseline_binary.predict(X_test) == y_test)),
                    sizes=[2, 16, 1], parameters=parameter_count(binary_parameters),
                    initial_loss=binary_loss[0], final_loss=binary_loss[-1],
                    accuracy=float(binary_accuracy)),
        multiclass=dict(train_counts=np.bincount(ym_train), test_counts=np.bincount(ym_test),
                        baseline_accuracy=float(np.mean(baseline_multi.predict(Xm_test) == ym_test)),
                        confusion_matrix=matrix, confusion_pair_counts=pair_counts),
        runs=runs, means=means,
        mean_accuracy_difference_pp=100 * (means[1]["accuracy"] - means[0]["accuracy"]))
    print(json.dumps(output, indent=2, ensure_ascii=False,
                     default=lambda value: value.tolist()))


if __name__ == "__main__":
    main()
