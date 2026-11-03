# Redes Neurais

Portfólio da disciplina de Redes Neurais.

## Atividade 1 — Dados

1. Baixe `train.csv` da competição [Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic/data).
2. Salve o arquivo em `data/spaceship-titanic/train.csv`.
3. Crie e ative o ambiente virtual:

   ```bash
   python3 -m venv env
   source env/bin/activate
   python3 -m pip install --upgrade pip
   python3 -m pip install -r requirements.txt
   ```

4. Gere números e figuras:

   ```bash
   python3 docs/exercises/data/code/data_exercise.py
   ```

5. Valide e visualize o site:

   ```bash
   mkdocs build --strict
   mkdocs serve
   ```

O arquivo bruto do Kaggle não deve ser enviado ao repositório. O script usa apenas
`train.csv`; `test.csv` e `sample_submission.csv` não fazem parte da atividade.

## Projeto — EDA Online Shoppers

Equipe: Daniel Marco Djanikian, Stefano Metne Lindenbojm e Felipe Frug Mauro.

O relatório está em `docs/project/eda/index.md`. O código calcula as tabelas,
valida o pipeline e gera oito figuras; não sobrescreve o relatório e não treina
classificadores. O CSV bruto da UCI já acompanha o repositório, sob licença
CC BY 4.0, com atribuição no relatório. Não é necessário baixar dados do Kaggle
para executar este projeto.

Em Python 3.12, a partir de um clone limpo:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
python docs/project/eda/code/eda.py > /tmp/eda-resultados.md
mkdocs build --strict
```

O arquivo `/tmp/eda-resultados.md` contém todas as tabelas numéricas em Markdown,
incluindo os dez resultados finais. As figuras ficam em
`docs/project/eda/figures/figura_01.png` a `figura_08.png`. A primeira execução
do UMAP pode levar mais tempo devido à compilação do Numba.

O script verifica o SHA-256 do CSV, a separação dos grupos, as estatísticas
aprendidas apenas no treino, o tratamento de ausências e categorias novas e a
ausência de NaN/inf. PCA, t-SNE e UMAP usam seed 42; as cinco projeções mostram
a mesma amostra estratificada de 2.000 sessões de treino. Para repetir a análise:

```bash
python docs/project/eda/code/eda.py > /tmp/eda-repeticao.md
diff -u /tmp/eda-resultados.md /tmp/eda-repeticao.md
PYTHONPATH=docs/project/eda/code python -c "import eda; print(eda.build_preprocessor())"
```

A reprodução deve usar o mesmo ambiente e versões. A importação apenas constrói
o objeto solicitado: não carrega dados nem gera figuras. Não se promete igualdade
binária das projeções entre plataformas ou versões distintas.

## Projeto — Classificação Online Shoppers

Relatório: `docs/project/classification/index.md`. O protocolo e os resultados
numéricos estão em `artifacts/`; pesos e intermediários são regenerados em
`.cache/classification/`, sem depender de GPU. O CSV da EDA já é versionado.

Em Python 3.12, a partir da raiz de um clone limpo:

```bash
unset PYTHONPATH
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
python docs/project/classification/code/train.py prepare
python -m unittest discover -s docs/project/classification/code -p 'test_*.py'
python docs/project/classification/code/train.py search
python docs/project/classification/code/train.py freeze
python docs/project/classification/code/evaluate.py run
python docs/project/classification/code/evaluate.py render
python -m unittest discover -s docs/project/classification/code -p 'test_*.py'
mkdocs build --strict
```

Os comandos `prepare`, `search` e `freeze` recusam sobrescrever etapas concluídas.
Para uma reprodução adicional, acrescente `--run-dir .cache/classification-reproduction`
a cada comando de treino/avaliação. Essa execução não sobrescreve os artefatos
publicados. `evaluate.py render` reconstrói gráficos usando predições salvas;
`evaluate.py run` não repete inferência se elas já existem. Não executar novas
buscas orientadas pelos resultados finais. `unset PYTHONPATH` evita importar
pacotes externos do ROS presentes no ambiente original de desenvolvimento.

A MLP obteve Macro-F1 de teste **0.638510 ± 0.006969**, contra **0.632041** da
regressão logística; o ganho **0.006469** não atingiu a meta pré-registrada de 0.02.
As evidências de reprodução estão em
`docs/project/classification/artifacts/verification.json`.
