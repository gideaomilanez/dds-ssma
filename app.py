"""
App Streamlit - Análise de DDS (SSMA) por Unidade e Regional

Este aplicativo lê a planilha de respostas do formulário de SSMA,
filtra os registros de DDS e gera automaticamente gráficos de:

1) Participação do regional (por regional)
2) Participação de regionais por unidade de trabalho
3) Participação de supervisores por regional
4) Participação dos supervisores por unidade
5) Percentual de DDS com supervisor presente por unidade
6) Percentual de DDS com supervisor presente por regional
7) Dias possíveis de DDS x realizados por unidade (barras sobrepostas)
8) Proporção de DDS realizados / dias possíveis por unidade

Regras importantes:
- Considera, no máximo, 1 DDS por dia/unidade (mantém o último registro do dia).
- Ignora registros da unidade de trabalho "COMERCIAL".
- Dias possíveis:
    * Log Arte / ArtePeças: segunda a sexta
    * Demais unidades (usinas): segunda a sábado
"""

import io
import unicodedata
from datetime import date
from typing import Dict, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# ============================================================
# CONFIGURAÇÃO STREAMLIT
# ============================================================
st.set_page_config(
    page_title="Análise de DDS - SSMA",
    layout="wide"
)

st.title("📊 Gerar gráficos SSMA")

st.markdown(
    """
Este app lê a planilha de **SSMA**, filtra os registros de **DDS** 
e gera automaticamente os principais gráficos de presença de **regionais** e **supervisores**.

> Obs.: O setor **COMERCIAL** é automaticamente filtrado da análise.
"""
)

# ============================================================
# CONFIGURAÇÃO: nomes EXATOS das colunas na planilha
# ============================================================
COL_FORMULARIO = "SELECIONE SEU FORMULÁRIO"
VALOR_DDS = "REALIZAÇÃO DE DDS - DIÁLOGO DIÁRIO DE SEGURANÇA"

COL_HORA_CONCLUSAO = "Hora de conclusão"
COL_REGIONAL = "REGIONAL"
COL_UNIDADE = "UNIDADE DE TRABALHO"

COL_PRES_REGIONAL = "a. PRESENÇA REGIONAL"
COL_PRES_SUPERVISOR = "a. PRESENÇA SUPERVISOR"

BAR_COLOR = "#082951"   # cor padrão das barras
BAR_RED = "#c0392b"     # cor para destaque negativo
BAR_GRAY = "#d3d3d3"    # cinza claro para "dias possíveis"


# ============================================================
# Funções auxiliares gerais
# ============================================================
def normalizar_texto(txt: str) -> str:
    """Remove acentos, deixa maiúsculo e tira espaços extras."""
    if not isinstance(txt, str):
        return ""
    txt = txt.strip()
    txt = unicodedata.normalize("NFD", txt)
    txt = "".join(ch for ch in txt if unicodedata.category(ch) != "Mn")
    return txt.upper()


def unidade_e_logarte_ou_artepecas(unidade: str) -> bool:
    """
    Retorna True se a unidade for Log Arte / ArtePeças (regime seg-sex),
    False caso contrário (regime seg-sáb).
    """
    norm = normalizar_texto(unidade)
    return (
        "LOGARTE" in norm
        or "LOG ARTE" in norm
        or "ARTEPECAS" in norm
        or "ARTE PECAS" in norm
    )


# ============================================================
# Função auxiliar: gráfico de barra com valores
# ============================================================
def plot_bar_with_labels(
    serie: pd.Series,
    title: str,
    xlabel: str,
    ylabel: str,
    rotation: int = 0,
    figsize=(12, 5),
    horizontal: bool = False,
    is_percent: bool = False,
    highlight_below: Optional[float] = None,
):
    """
    Gera um gráfico de barras (horizontal ou vertical) com rótulos numéricos nas barras.

    Se highlight_below não for None, pinta de vermelho as barras com valor abaixo do limite.
    Retorna:
        fig (matplotlib.figure.Figure): figura pronta para exibição/salvamento.
    """
    # garante série numérica onde fizer sentido
    serie = pd.to_numeric(serie, errors="coerce")

    # valores numéricos limpos (substitui NaN/Inf por 0)
    vals = pd.to_numeric(serie.values, errors="coerce")
    vals = np.where(np.isfinite(vals), vals, 0.0)

    # define cores (vermelho para valores abaixo do limite, se configurado)
    if highlight_below is not None:
        colors = [
            BAR_RED if v < highlight_below else BAR_COLOR
            for v in vals
        ]
    else:
        colors = BAR_COLOR

    fig, ax = plt.subplots(figsize=figsize)

    # Plot das barras
    if horizontal:
        serie.plot(kind="barh", ax=ax, color=colors)
    else:
        serie.plot(kind="bar", ax=ax, color=colors)

    # Título e labels dos eixos
    ax.set_title(title)
    if horizontal:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    else:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=rotation)

    # ---------------- DESPINE / ESTILO ----------------
    if horizontal:
        # UNIDADES (barras horizontais):
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)

        # remove ticks e números do eixo X
        ax.tick_params(axis="x", which="both", length=0)
        ax.set_xticks([])
        ax.set_xticklabels([])
    else:
        # REGIONAIS (barras verticais):
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.tick_params(axis="y", which="both", length=0)
        ax.tick_params(axis="x", which="both", length=0)
    # --------------------------------------------------

    # Adiciona os valores nas barras
    if horizontal:
        if len(vals) == 0:
            max_val = 1.0
        else:
            max_val = float(vals.max())
            if max_val <= 0:
                max_val = 1.0

        desloc = max_val * 0.02
        ax.set_xlim(0, max_val * 1.15)

        for i, v in enumerate(vals):
            label = f"{v:.1f} %" if is_percent else (
                str(int(v)) if float(v).is_integer() else f"{v:.1f}"
            )
            ax.text(
                v + desloc,
                i,
                label,
                va="center",
                ha="left",
                fontsize=10,
            )
    else:
        for i, v in enumerate(vals):
            label = f"{v:.1f} %" if is_percent else (
                str(int(v)) if float(v).is_integer() else f"{v:.1f}"
            )
            ax.text(
                i,
                v,
                label,
                ha="center",
                va="bottom",
                fontsize=10,
            )

    plt.tight_layout()
    return fig


# ============================================================
# Gráfico especial: dias possíveis x realizados (barras sobrepostas)
# ============================================================
def plot_dds_potencial_real(
    possiveis: pd.Series,
    realizados: pd.Series,
    title: str,
    figsize=(12, 5),
    highlight_threshold: float = 69.0,
):
    """
    Gráfico horizontal de barras sobrepostas:
    - base cinza claro = dias possíveis
    - barra sobreposta azul/vermelha = dias realizados
    Cor vermelha quando (realizados / possíveis) * 100 < highlight_threshold.
    """
    # Alinha índices
    realizados = realizados.reindex(possiveis.index).fillna(0)

    poss_vals = pd.to_numeric(possiveis.values, errors="coerce")
    poss_vals = np.where(np.isfinite(poss_vals), poss_vals, 0.0)

    real_vals = pd.to_numeric(realizados.values, errors="coerce")
    real_vals = np.where(np.isfinite(real_vals), real_vals, 0.0)

    # Percentuais para definir cor
    with np.errstate(divide="ignore", invalid="ignore"):
        perc = np.where(poss_vals > 0, (real_vals / poss_vals) * 100, 0.0)

    colors_real = [
        BAR_RED if p < highlight_threshold else BAR_COLOR
        for p in perc
    ]

    fig, ax = plt.subplots(figsize=figsize)

    y_pos = np.arange(len(possiveis.index))

    max_val = poss_vals.max() if len(poss_vals) else 1.0
    if max_val <= 0:
        max_val = 1.0

    # Barra de fundo (dias possíveis)
    ax.barh(y_pos, poss_vals, color=BAR_GRAY, edgecolor="none")

    # Barra de realizados por cima
    ax.barh(y_pos, real_vals, color=colors_real, edgecolor="none")

    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("")

    # Despine: tira topo, direita e base
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    # Sem ticks no eixo X
    ax.tick_params(axis="x", which="both", length=0)
    ax.set_xticks([])
    ax.set_xticklabels([])

    # Nomes das unidades no eixo Y
    ax.set_yticks(y_pos)
    ax.set_yticklabels(possiveis.index)

    # Labels "realizados / possíveis"
    desloc = max_val * 0.02
    ax.set_xlim(0, max_val * 1.15)

    for i, (r, p) in enumerate(zip(real_vals, poss_vals)):
        label = f"{int(r)}/{int(p)}"
        ax.text(
            r + desloc,
            i,
            label,
            va="center",
            ha="left",
            fontsize=10,
        )

    plt.tight_layout()
    return fig


# ============================================================
# Função auxiliar: salva figura em buffer para download
# ============================================================
def figure_to_png_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()


# ============================================================
# Função principal de análise (lógica, sem Streamlit dentro)
# ============================================================
def analisar_dds(
    df: pd.DataFrame,
    data_ini: Optional[date] = None,
    data_fim: Optional[date] = None,
) -> Tuple[pd.DataFrame, Dict[str, plt.Figure], date, date]:
    """
    Processa o DataFrame original e retorna:

        - dds_unico: DataFrame com 1 DDS por dia/unidade
        - figs: dicionário {nome_grafico: figura_matplotlib}
        - data_ini_efetiva, data_fim_efetiva: intervalo efetivamente utilizado
    """
    # 1) Valida colunas básicas
    obrigatorias = [
        COL_FORMULARIO,
        COL_HORA_CONCLUSAO,
        COL_REGIONAL,
        COL_UNIDADE,
        COL_PRES_REGIONAL,
        COL_PRES_SUPERVISOR,
    ]
    for col in obrigatorias:
        if col not in df.columns:
            raise ValueError(f"Coluna obrigatória não encontrada na planilha: '{col}'")

    # 2) Filtra COMERCIAL
    df = df[df[COL_UNIDADE] != "COMERCIAL"].copy()
    if df.empty:
        raise ValueError("Após filtrar UNIDADE DE TRABALHO = 'COMERCIAL', não sobraram registros.")

    # 3) Prepara datas e filtra DDS
    df[COL_HORA_CONCLUSAO] = pd.to_datetime(df[COL_HORA_CONCLUSAO])
    df["data"] = df[COL_HORA_CONCLUSAO].dt.date

    dds = df[df[COL_FORMULARIO] == VALOR_DDS].copy()
    if dds.empty:
        raise ValueError(
            f"Não foram encontrados registros de DDS com o valor "
            f"'{VALOR_DDS}' na coluna '{COL_FORMULARIO}'."
        )

    # 4) Define período padrão se necessário
    if data_ini is None:
        data_ini = dds["data"].min()
    if data_fim is None:
        data_fim = dds["data"].max()

    data_ini = pd.to_datetime(data_ini).date()
    data_fim = pd.to_datetime(data_fim).date()

    mascara = (dds["data"] >= data_ini) & (dds["data"] <= data_fim)
    dds_periodo = dds.loc[mascara].copy()
    if dds_periodo.empty:
        raise ValueError(f"Não há DDS no período selecionado ({data_ini} a {data_fim}).")

    # 5) 1 DDS por dia/unidade (mantém o último registro do dia)
    dds_periodo = dds_periodo.sort_values(COL_HORA_CONCLUSAO)
    dds_unico = dds_periodo.drop_duplicates(subset=["data", COL_UNIDADE], keep="last").copy()

    # 6) Flags de presença
    dds_unico["regional_presente"] = dds_unico[COL_PRES_REGIONAL].eq("SIM")
    dds_unico["supervisor_presente"] = dds_unico[COL_PRES_SUPERVISOR].eq("SIM")

    figs: Dict[str, plt.Figure] = {}

    # ---------------- Gráfico 1: Regional por REGIONAL ----------------
    df_regional_presente = dds_unico[dds_unico["regional_presente"]]
    if not df_regional_presente.empty:
        freq_regional_por_regional = (
            df_regional_presente.groupby(COL_REGIONAL).size().sort_values(ascending=False)
        )
        figs["participacao_regional_por_regional"] = plot_bar_with_labels(
            freq_regional_por_regional,
            title="Participação do regional",
            xlabel="",
            ylabel="",
            rotation=0,
            figsize=(8, 4),
            horizontal=False,
        )

    # ---------------- Gráfico 2: Regional por UNIDADE -----------------
    if not df_regional_presente.empty:
        freq_regional_por_unidade = (
            df_regional_presente.groupby(COL_UNIDADE).size().sort_values(ascending=False)
        )
        figs["participacao_regional_por_unidade"] = plot_bar_with_labels(
            freq_regional_por_unidade,
            title="Participação de regionais por unidade de trabalho",
            xlabel="",
            ylabel="",
            rotation=0,
            figsize=(10, max(5, len(freq_regional_por_unidade) * 0.3)),
            horizontal=True,
        )

    # ---------------- Gráfico 3: Supervisor por REGIONAL --------------
    df_supervisor_presente = dds_unico[dds_unico["supervisor_presente"]]
    if not df_supervisor_presente.empty:
        freq_supervisor_por_regional = (
            df_supervisor_presente.groupby(COL_REGIONAL).size().sort_values(ascending=False)
        )
        figs["participacao_supervisor_por_regional"] = plot_bar_with_labels(
            freq_supervisor_por_regional,
            title="Participação de supervisores por regional",
            xlabel="",
            ylabel="",
            rotation=0,
            figsize=(8, 4),
            horizontal=False,
        )

    # ---------------- Gráfico 4: Supervisor por UNIDADE ---------------
    if not df_supervisor_presente.empty:
        freq_supervisor_por_unidade = (
            df_supervisor_presente.groupby(COL_UNIDADE).size().sort_values(ascending=False)
        )
        figs["participacao_supervisor_por_unidade"] = plot_bar_with_labels(
            freq_supervisor_por_unidade,
            title="Participação dos supervisores",
            xlabel="",
            ylabel="",
            rotation=0,
            figsize=(10, max(5, len(freq_supervisor_por_unidade) * 0.3)),
            horizontal=True,
        )

    # ---------------- Gráfico 5: % Supervisor por UNIDADE -------------
    total_dds_por_unidade = dds_unico.groupby(COL_UNIDADE).size()
    sup_dds_por_unidade = (
        dds_unico[dds_unico["supervisor_presente"]].groupby(COL_UNIDADE).size()
    )
    percentual_sup_por_unidade = (
        (sup_dds_por_unidade / total_dds_por_unidade * 100)
        .reindex(total_dds_por_unidade.index, fill_value=0)
        .astype(float)
        .sort_values(ascending=False)
    )
    if not percentual_sup_por_unidade.empty:
        figs["percentual_supervisor_por_unidade"] = plot_bar_with_labels(
            percentual_sup_por_unidade,
            title="Percentual de presença dos supervisores por unidade",
            xlabel="",
            ylabel="",
            rotation=0,
            figsize=(10, max(5, len(percentual_sup_por_unidade) * 0.3)),
            horizontal=True,
            is_percent=True,
            highlight_below=60.0,  # corte de 60% para presença de supervisor
        )

    # ---------------- Gráfico 6: % Supervisor por REGIONAL ------------
    total_dds_por_regional = dds_unico.groupby(COL_REGIONAL).size()
    sup_dds_por_regional = (
        dds_unico[dds_unico["supervisor_presente"]].groupby(COL_REGIONAL).size()
    )
    percentual_sup_por_regional = (
        (sup_dds_por_regional / total_dds_por_regional * 100)
        .reindex(total_dds_por_regional.index, fill_value=0)
        .astype(float)
        .sort_values(ascending=False)
    )
    if not percentual_sup_por_regional.empty:
        figs["percentual_supervisor_por_regional"] = plot_bar_with_labels(
            percentual_sup_por_regional,
            title="Percentual de presença dos supervisores por regional",
            xlabel="",
            ylabel="",
            rotation=0,
            figsize=(8, 4),
            horizontal=False,
            is_percent=True,
        )

    # ============================================================
    # 7 e 8) Potencial de DDS (dias possíveis x realizados)
    # ============================================================

    # DDS realizados por unidade (já está 1 por dia/unidade)
    dds_realizados_por_unidade = dds_unico.groupby(COL_UNIDADE).size()

    # Intervalo completo de dias
    todos_os_dias = pd.date_range(data_ini, data_fim, freq="D")
    dias_seg_a_sex = todos_os_dias[todos_os_dias.weekday < 5].size  # 0-4
    dias_seg_a_sab = todos_os_dias[todos_os_dias.weekday < 6].size  # 0-5

    # Dias possíveis por unidade
    unidades_idx = dds_realizados_por_unidade.index
    possiveis_dds_vals = []
    for unidade in unidades_idx:
        if unidade_e_logarte_ou_artepecas(unidade):
            possiveis_dds_vals.append(dias_seg_a_sex)
        else:
            possiveis_dds_vals.append(dias_seg_a_sab)

    possiveis_dds_por_unidade = pd.Series(
        possiveis_dds_vals,
        index=unidades_idx,
        name="dias_possiveis_dds",
    )

    # Ordena dias possíveis x realizados pela quantidade REALIZADA (decrescente)
    dds_realizados_sorted = dds_realizados_por_unidade.sort_values(ascending=False)
    possiveis_alinhado = possiveis_dds_por_unidade.reindex(dds_realizados_sorted.index)

    figs["dias_possiveis_x_realizados_por_unidade"] = plot_dds_potencial_real(
        possiveis_alinhado,
        dds_realizados_sorted,
        title="Dias realizados de DDS x dias possíveis",
        figsize=(10, max(5, len(possiveis_alinhado) * 0.3)),
        highlight_threshold=60.0,
    )

    # Proporção de DDS realizados sobre dias possíveis
    with np.errstate(divide="ignore", invalid="ignore"):
        proporcao_dds_por_unidade = (
            (dds_realizados_sorted / possiveis_alinhado) * 100
        )

    proporcao_dds_por_unidade = (
        proporcao_dds_por_unidade.replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .astype(float)
        .sort_values(ascending=False)
    )

    figs["proporcao_dds_realizados_por_unidade"] = plot_bar_with_labels(
        proporcao_dds_por_unidade,
        title="Proporção de DDS realizados em relação aos dias possíveis",
        xlabel="",
        ylabel="",
        rotation=0,
        figsize=(10, max(5, len(proporcao_dds_por_unidade) * 0.3)),
        horizontal=True,
        is_percent=True,
        highlight_below=60.0,  # <60% em vermelho
    )

    return dds_unico, figs, data_ini, data_fim


# ============================================================
# INTERFACE STREAMLIT
# ============================================================
st.sidebar.header("⚙️ Configurações")

uploaded_file = st.sidebar.file_uploader(
    "1) Envie o arquivo Excel de SSMA",
    type=["xlsx", "xls"],
    help="Exportação do formulário de SSMA contendo os registros de DDS.",
)

data_ini_input: Optional[date] = None
data_fim_input: Optional[date] = None

if uploaded_file is not None:
    # Lê arquivo
    try:
        df_raw = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Erro ao ler o arquivo: {e}")
        st.stop()

    st.sidebar.success(f"Arquivo carregado: {df_raw.shape[0]} linhas, {df_raw.shape[1]} colunas.")

    # Sugere período padrão com base nos dados
    if COL_HORA_CONCLUSAO in df_raw.columns:
        try:
            datas_tmp = pd.to_datetime(df_raw[COL_HORA_CONCLUSAO], errors="coerce").dt.date
            min_data = datas_tmp.min()
            max_data = datas_tmp.max()
        except Exception:
            min_data, max_data = None, None
    else:
        min_data, max_data = None, None

    data_ini_input = st.sidebar.date_input(
        "2) Data inicial",
        value=min_data if isinstance(min_data, date) else date.today(),
    )
    data_fim_input = st.sidebar.date_input(
        "3) Data final",
        value=max_data if isinstance(max_data, date) else date.today(),
    )

    gerar = st.sidebar.button("Gerar gráficos")

    if gerar:
        if data_ini_input is None or data_fim_input is None:
            st.warning("Informe data inicial e data final.")
            st.stop()

        with st.spinner("Processando dados e gerando gráficos..."):
            try:
                dds_unico, figs, data_ini_efetiva, data_fim_efetiva = analisar_dds(
                    df_raw,
                    data_ini=data_ini_input,
                    data_fim=data_fim_input,
                )
            except ValueError as e:
                st.error(str(e))
                st.stop()

        st.success(
            f"Análise concluída! Período considerado: **{data_ini_efetiva}** a **{data_fim_efetiva}**. "
            f"Total de registros únicos (1 por dia/unidade): **{len(dds_unico)}**."
        )

        st.subheader("📈 Gráficos gerados")

        for nome, fig in figs.items():
            titulo = fig.axes[0].get_title()
            st.markdown(f"### {titulo}")
            st.pyplot(fig)

            png_bytes = figure_to_png_bytes(fig)
            st.download_button(
                label="⬇️ Baixar este gráfico em PNG",
                data=png_bytes,
                file_name=f"{nome}.png",
                mime="image/png",
            )

            plt.close(fig)

else:
    st.info("Use o menu lateral para enviar a planilha e configurar o período de análise.")
