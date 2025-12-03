"""
App Streamlit - Análise de DDS (SSMA) por Unidade e Regional

Este aplicativo lê a planilha de respostas do formulário de SSMA,
filtra os registros de DDS e gera automaticamente gráficos de:

1) Participação do regional (por regional)
2) Participação de regionais por unidade de trabalho
3) Participação de supervisores por regional
4) Participação dos supervisores por unidade (contagem, camadas)
   -> dias com DDS + supervisor / dias de DDS / dias possíveis
5) Participação dos supervisores por unidade (percentual, camadas)
   -> percentual de dias com DDS + supervisor em relação aos dias possíveis
6) Percentual de DDS com supervisor presente por unidade (barras simples)
7) Percentual de DDS com supervisor presente por regional
8) Dias possíveis de DDS x realizados por unidade (barras sobrepostas)
9) Proporção de DDS realizados / dias possíveis por unidade

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

BAR_COLOR = "#082951"      # azul padrão
BAR_RED = "#c0392b"        # vermelho corte
BAR_GRAY = "#d0d4d6"       # cinza claro (dias possíveis)
BAR_DARK_GRAY = "#b0b7ba"  # cinza levemente mais escuro (dias com DDS)


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
    True se a unidade for Log Arte / ArtePeças (regime seg-sex),
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
# Gráfico simples: barra com valores
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
    """Gráfico simples com valores na barra."""
    serie = pd.to_numeric(serie, errors="coerce")
    vals = pd.to_numeric(serie.values, errors="coerce")
    vals = np.where(np.isfinite(vals), vals, 0.0)

    if highlight_below is not None:
        colors = [BAR_RED if v < highlight_below else BAR_COLOR for v in vals]
    else:
        colors = BAR_COLOR

    fig, ax = plt.subplots(figsize=figsize)

    if horizontal:
        serie.plot(kind="barh", ax=ax, color=colors)
    else:
        serie.plot(kind="bar", ax=ax, color=colors)

    ax.set_title(title)
    if horizontal:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    else:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=rotation)

    if horizontal:
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.tick_params(axis="x", which="both", length=0)
        ax.set_xticks([])
        ax.set_xticklabels([])
    else:
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.tick_params(axis="y", which="both", length=0)
        ax.tick_params(axis="x", which="both", length=0)

    if horizontal:
        max_val = max(vals.max(), 1.0) if len(vals) else 1.0
        if max_val <= 0:
            max_val = 1.0
        desloc = max_val * 0.02
        ax.set_xlim(0, max_val * 1.15)

        for i, v in enumerate(vals):
            label = f"{v:.1f} %" if is_percent else (
                str(int(v)) if float(v).is_integer() else f"{v:.1f}"
            )
            ax.text(v + desloc, i, label, va="center", ha="left", fontsize=10)
    else:
        for i, v in enumerate(vals):
            label = f"{v:.1f} %" if is_percent else (
                str(int(v)) if float(v).is_integer() else f"{v:.1f}"
            )
            ax.text(i, v, label, ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    return fig


# ============================================================
# Gráfico: dias possíveis x DDS realizados (geral)
# ============================================================
def plot_dds_potencial_real(
    possiveis: pd.Series,
    realizados: pd.Series,
    title: str,
    figsize=(12, 5),
    highlight_threshold: float = 60.0,
):
    """Cinza claro = possíveis; azul/vermelho = DDS realizados."""
    realizados = realizados.reindex(possiveis.index).fillna(0)

    poss_vals = pd.to_numeric(possiveis.values, errors="coerce")
    poss_vals = np.where(np.isfinite(poss_vals), poss_vals, 0.0)

    real_vals = pd.to_numeric(realizados.values, errors="coerce")
    real_vals = np.where(np.isfinite(real_vals), real_vals, 0.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        perc = np.where(poss_vals > 0, (real_vals / poss_vals) * 100, 0.0)

    colors_real = [BAR_RED if p < highlight_threshold else BAR_COLOR for p in perc]

    fig, ax = plt.subplots(figsize=figsize)
    y_pos = np.arange(len(possiveis.index))

    max_val = poss_vals.max() if len(poss_vals) else 1.0
    if max_val <= 0:
        max_val = 1.0

    ax.barh(y_pos, poss_vals, color=BAR_GRAY, edgecolor="none")
    ax.barh(y_pos, real_vals, color=colors_real, edgecolor="none")

    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("")

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    ax.tick_params(axis="x", which="both", length=0)
    ax.set_xticks([])
    ax.set_xticklabels([])

    ax.set_yticks(y_pos)
    ax.set_yticklabels(possiveis.index)

    desloc = max_val * 0.02
    ax.set_xlim(0, max_val * 1.15)

    for i, (r, p) in enumerate(zip(real_vals, poss_vals)):
        label = f"{int(r)}/{int(p)}"
        ax.text(r + desloc, i, label, va="center", ha="left", fontsize=10)

    plt.tight_layout()
    return fig


# ============================================================
# Gráfico camadas SUPERVISOR: contagem
# ============================================================
def plot_supervisor_potencial_real_counts(
    possiveis: pd.Series,
    dds_realizados: pd.Series,
    dias_com_supervisor: pd.Series,
    title: str,
    figsize=(12, 5),
    highlight_threshold: float = 60.0,
):
    """
    Cinza claro = dias possíveis
    Cinza escuro = dias com DDS
    Azul/Vermelho = dias com DDS + supervisor

    Rótulo numérico: contagem "supervisor / possíveis".
    """
    dds_realizados = dds_realizados.reindex(possiveis.index).fillna(0)
    dias_com_supervisor = dias_com_supervisor.reindex(possiveis.index).fillna(0)

    poss_vals = np.where(
        np.isfinite(pd.to_numeric(possiveis.values, errors="coerce")), 
        pd.to_numeric(possiveis.values, errors="coerce"),
        0.0
    )
    dds_vals = np.where(
        np.isfinite(pd.to_numeric(dds_realizados.values, errors="coerce")),
        pd.to_numeric(dds_realizados.values, errors="coerce"),
        0.0
    )
    sup_vals = np.where(
        np.isfinite(pd.to_numeric(dias_com_supervisor.values, errors="coerce")),
        pd.to_numeric(dias_com_supervisor.values, errors="coerce"),
        0.0
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        perc_sup_raw = np.where(poss_vals > 0, (sup_vals / poss_vals) * 100, 0.0)
    perc_sup = np.where(np.isfinite(perc_sup_raw), perc_sup_raw, 0.0)

    colors_sup = [BAR_RED if p < highlight_threshold else BAR_COLOR for p in perc_sup]

    fig, ax = plt.subplots(figsize=figsize)
    y_pos = np.arange(len(possiveis.index))

    max_val = poss_vals.max() if len(poss_vals) else 1.0
    if max_val <= 0:
        max_val = 1.0

    ax.barh(y_pos, poss_vals, color=BAR_GRAY, edgecolor="none")
    ax.barh(y_pos, dds_vals, color=BAR_DARK_GRAY, edgecolor="none")
    ax.barh(y_pos, sup_vals, color=colors_sup, edgecolor="none")

    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("")

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    ax.tick_params(axis="x", which="both", length=0)
    ax.set_xticks([])
    ax.set_xticklabels([])

    ax.set_yticks(y_pos)
    ax.set_yticklabels(possiveis.index)

    desloc = max_val * 0.02
    ax.set_xlim(0, max_val * 1.15)

    # rótulo: supervisor / possíveis (contagem)
    for i, (s, p) in enumerate(zip(sup_vals, poss_vals)):
        label = f"{int(s)}/{int(p)}"
        ax.text(s + desloc, i, label, va="center", ha="left", fontsize=10)

    plt.tight_layout()
    return fig


# ============================================================
# Gráfico camadas SUPERVISOR: percentual
# ============================================================
def plot_supervisor_potencial_real(
    possiveis: pd.Series,
    dds_realizados: pd.Series,
    dias_com_supervisor: pd.Series,
    title: str,
    figsize=(12, 5),
    highlight_threshold: float = 60.0,
):
    """
    Igual ao anterior, mas o rótulo é só o percentual de participação
    do supervisor em relação aos dias possíveis.
    """
    dds_realizados = dds_realizados.reindex(possiveis.index).fillna(0)
    dias_com_supervisor = dias_com_supervisor.reindex(possiveis.index).fillna(0)

    poss_vals = np.where(
        np.isfinite(pd.to_numeric(possiveis.values, errors="coerce")), 
        pd.to_numeric(possiveis.values, errors="coerce"),
        0.0
    )
    dds_vals = np.where(
        np.isfinite(pd.to_numeric(dds_realizados.values, errors="coerce")),
        pd.to_numeric(dds_realizados.values, errors="coerce"),
        0.0
    )
    sup_vals = np.where(
        np.isfinite(pd.to_numeric(dias_com_supervisor.values, errors="coerce")),
        pd.to_numeric(dias_com_supervisor.values, errors="coerce"),
        0.0
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        perc_sup_raw = np.where(poss_vals > 0, (sup_vals / poss_vals) * 100, 0.0)
    perc_sup = np.where(np.isfinite(perc_sup_raw), perc_sup_raw, 0.0)

    colors_sup = [BAR_RED if p < highlight_threshold else BAR_COLOR for p in perc_sup]

    fig, ax = plt.subplots(figsize=figsize)
    y_pos = np.arange(len(possiveis.index))

    max_val = poss_vals.max() if len(poss_vals) else 1.0
    if max_val <= 0:
        max_val = 1.0

    ax.barh(y_pos, poss_vals, color=BAR_GRAY, edgecolor="none")
    ax.barh(y_pos, dds_vals, color=BAR_DARK_GRAY, edgecolor="none")
    ax.barh(y_pos, sup_vals, color=colors_sup, edgecolor="none")

    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("")

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    ax.tick_params(axis="x", which="both", length=0)
    ax.set_xticks([])
    ax.set_xticklabels([])

    ax.set_yticks(y_pos)
    ax.set_yticklabels(possiveis.index)

    desloc = max_val * 0.02
    ax.set_xlim(0, max_val * 1.15)

    # rótulo: somente o percentual do supervisor
    for i, (s, perc) in enumerate(zip(sup_vals, perc_sup)):
        label = f"{perc:.1f} %"
        ax.text(s + desloc, i, label, va="center", ha="left", fontsize=10)

    plt.tight_layout()
    return fig


# ============================================================
# Auxiliar: fig -> PNG
# ============================================================
def figure_to_png_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()


# ============================================================
# Lógica principal
# ============================================================
def analisar_dds(
    df: pd.DataFrame,
    data_ini: Optional[date] = None,
    data_fim: Optional[date] = None,
) -> Tuple[pd.DataFrame, Dict[str, plt.Figure], date, date]:
    # valida colunas
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

    # filtra COMERCIAL
    df = df[df[COL_UNIDADE] != "COMERCIAL"].copy()
    if df.empty:
        raise ValueError("Após filtrar UNIDADE DE TRABALHO = 'COMERCIAL', não sobraram registros.")

    df[COL_HORA_CONCLUSAO] = pd.to_datetime(df[COL_HORA_CONCLUSAO])
    df["data"] = df[COL_HORA_CONCLUSAO].dt.date

    dds = df[df[COL_FORMULARIO] == VALOR_DDS].copy()
    if dds.empty:
        raise ValueError(
            f"Não foram encontrados registros de DDS com o valor "
            f"'{VALOR_DDS}' na coluna '{COL_FORMULARIO}'."
        )

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

    # mantém o último registro do dia/unidade
    dds_periodo = dds_periodo.sort_values(COL_HORA_CONCLUSAO)
    dds_unico = dds_periodo.drop_duplicates(subset=["data", COL_UNIDADE], keep="last").copy()

    dds_unico["regional_presente"] = dds_unico[COL_PRES_REGIONAL].eq("SIM")
    dds_unico["supervisor_presente"] = dds_unico[COL_PRES_SUPERVISOR].eq("SIM")

    # dias possíveis por unidade
    todos_os_dias = pd.date_range(data_ini, data_fim, freq="D")
    dias_seg_a_sex = todos_os_dias[todos_os_dias.weekday < 5].size
    dias_seg_a_sab = todos_os_dias[todos_os_dias.weekday < 6].size

    unidades_idx = np.sort(dds_unico[COL_UNIDADE].unique())
    possiveis_dds_por_unidade = pd.Series(
        {
            unidade: (
                dias_seg_a_sex
                if unidade_e_logarte_ou_artepecas(unidade)
                else dias_seg_a_sab
            )
            for unidade in unidades_idx
        }
    )

    dds_realizados_por_unidade = dds_unico.groupby(COL_UNIDADE).size()

    figs: Dict[str, plt.Figure] = {}

    # 1) regional por regional
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

    # 2) regional por unidade
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

    # 3) supervisor por regional
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

    # 4) supervisor por unidade - CONTAGEM (camadas)
    if not df_supervisor_presente.empty:
        sup_dds_por_unidade = (
            df_supervisor_presente.groupby(COL_UNIDADE).size()
        )
        sup_sorted = sup_dds_por_unidade.sort_values(ascending=False)

        poss_sup = possiveis_dds_por_unidade.reindex(sup_sorted.index)
        dds_sup = dds_realizados_por_unidade.reindex(sup_sorted.index)

        figs["participacao_supervisor_por_unidade_contagem"] = plot_supervisor_potencial_real_counts(
            poss_sup,
            dds_sup,
            sup_sorted,
            title=(
                "Participação dos supervisores vs dias possíveis de DDS"
            ),
            figsize=(10, max(5, len(poss_sup) * 0.3)),
            highlight_threshold=60.0,
        )

        # 5) supervisor por unidade - PERCENTUAL (camadas)
        figs["participacao_supervisor_por_unidade_percentual_camadas"] = plot_supervisor_potencial_real(
            poss_sup,
            dds_sup,
            sup_sorted,
            title=(
                "Percentual de participação dos supervisores vs dias possíveis de DDS"
            ),
            figsize=(10, max(5, len(poss_sup) * 0.3)),
            highlight_threshold=60.0,
        )


    # 7) % supervisor por regional
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

    # 8) dias possíveis x DDS realizados
    dds_realizados_sorted = dds_realizados_por_unidade.sort_values(ascending=False)
    possiveis_alinhado = possiveis_dds_por_unidade.reindex(dds_realizados_sorted.index)

    figs["dias_possiveis_x_realizados_por_unidade"] = plot_dds_potencial_real(
        possiveis_alinhado,
        dds_realizados_sorted,
        title="Dias realizados de DDS x dias possíveis por unidade",
        figsize=(10, max(5, len(possiveis_alinhado) * 0.3)),
        highlight_threshold=60.0,
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
    try:
        df_raw = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Erro ao ler o arquivo: {e}")
        st.stop()

    st.sidebar.success(f"Arquivo carregado: {df_raw.shape[0]} linhas, {df_raw.shape[1]} colunas.")

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
