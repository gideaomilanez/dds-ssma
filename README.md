# 📊 Análise de DDS

Aplicativo em **Python** para análise de registros de **DDS (Diálogo Diário de Segurança)** a partir da planilha de respostas do formulário de **SSMA**.

## ✨ Funcionalidades

- Upload de planilha Excel (`.xlsx` / `.xls`)
- Filtro por intervalo de datas (com base na coluna `Hora de conclusão`)
- Regras automáticas:
  - Considera no máximo **1 DDS por dia/unidade** (mantém o último registro do dia)
  - Filtra automaticamente a unidade de trabalho **COMERCIAL**
- Geração automática de 6 gráficos:
  1. Participação do regional (por regional)
  2. Participação de regionais por unidade de trabalho
  3. Participação de supervisores por regional
  4. Participação dos supervisores por unidade
  5. Percentual de DDS com supervisor presente por unidade
  6. Percentual de DDS com supervisor presente por regional
- Download de cada gráfico em PNG diretamente pela interface

## 🛠️ Tecnologias

- Python
- [Streamlit](https://streamlit.io/)
- Pandas
- Matplotlib
- Numpy

## 📦 Instalação

Clone o repositório e instale as dependências:

```bash
pip install -r requirements.txt
