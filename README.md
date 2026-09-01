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
