import io
import unicodedata
from datetime import date
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# -------------------------------
# Constantes das colunas
# -------------------------------
COL_FORMULARIO = "SELECIONE SEU FORMULÁRIO"
VALOR_DDS = "REALIZAÇÃO DE DDS - DIÁLOGO DIÁRIO DE SEGURANÇA"
COL_HORA_CONCLUSAO = "Hora de conclusão"
COL_REGIONAL = "REGIONAL"
COL_UNIDADE = "UNIDADE DE TRABALHO"
COL_PRES_REGIONAL = "a. PRESENÇA REGIONAL"
COL_PRES_SUPERVISOR = "a. PRESENÇA SUPERVISOR"

# Cores
BAR_COLOR = "#082951"
BAR_RED = "#c0392b"
BAR_GRAY = "#d0d4d6"
BAR_DARK_GRAY = "#b0b7ba"
BAR_GREEN = "#27ae60"


# -------------------------------
# Funções auxiliares
# -------------------------------
def normalizar_texto(txt: str) -> str:
    if not isinstance(txt, str):
        return ""
    txt = txt.strip()
    txt = unicodedata.normalize("NFD", txt)
    txt = "".join(ch for ch in txt if unicodedata.category(ch) != "Mn")
    return txt.upper()


def eh_logarte_ou_artepecas(txt: str) -> bool:
    norm = normalizar_texto(txt)
    return (
        "LOGARTE" in norm
        or "LOG ARTE" in norm
        or "ARTEPECAS" in norm
        or "ARTE PECAS" in norm
    )


def unidade_e_logarte_ou_artepecas(unidade: str) -> bool:
    return eh_logarte_ou_artepecas(unidade)


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
    """Gráfico de barra simples com rótulos."""
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


def plot_dds_potencial_real(
    possiveis: pd.Series,
    realizados: pd.Series,
    title: str,
    figsize=(12, 5),
    highlight_threshold: float = 60.0,
):
    """
    Barras horizontais:
      - cinza claro: dias possíveis
      - verde/vermelho: dias realizados (% sobre possíveis)
      - rótulo: realizado/possível
    """
    realizados = realizados.reindex(possiveis.index).fillna(0)

    poss_vals = pd.to_numeric(possiveis.values, errors="coerce")
    poss_vals = np.where(np.isfinite(poss_vals), poss_vals, 0.0)

    real_vals = pd.to_numeric(realizados.values, errors="coerce")
    real_vals = np.where(np.isfinite(real_vals), real_vals, 0.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        perc = np.where(poss_vals > 0, (real_vals / poss_vals) * 100, 0.0)

    colors_real = [BAR_RED if p < highlight_threshold else BAR_GREEN for p in perc]

    fig, ax = plt.subplots(figsize=figsize)
    y_pos = np.arange(len(possiveis.index))

    max_val = poss_vals.max() if len(possiveis.index) else 1.0
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


def plot_supervisor_potencial_real_counts(
    possiveis: pd.Series,
    dds_realizados: pd.Series,
    dias_com_lideranca: pd.Series,
    title: str,
    figsize=(12, 5),
    highlight_threshold: float = 60.0,
):
    """
    Contagem:
      - cinza claro: dias possíveis
      - cinza escuro: dias com DDS
      - verde/vermelho: dias com liderança (sup / R1 / R2 / LogArte/ArtePeças / GT)
    Rótulo = liderança/possíveis.
    """
    dds_realizados = dds_realizados.reindex(possiveis.index).fillna(0)
    dias_com_lideranca = dias_com_lideranca.reindex(possiveis.index).fillna(0)

    poss_vals = np.where(
        np.isfinite(pd.to_numeric(possiveis.values, errors="coerce")),
        pd.to_numeric(possiveis.values, errors="coerce"),
        0.0,
    )
    dds_vals = np.where(
        np.isfinite(pd.to_numeric(dds_realizados.values, errors="coerce")),
        pd.to_numeric(dds_realizados.values, errors="coerce"),
        0.0,
    )
    lid_vals = np.where(
        np.isfinite(pd.to_numeric(dias_com_lideranca.values, errors="coerce")),
        pd.to_numeric(dias_com_lideranca.values, errors="coerce"),
        0.0,
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        perc_lid_raw = np.where(poss_vals > 0, (lid_vals / poss_vals) * 100, 0.0)
    perc_lid = np.where(np.isfinite(perc_lid_raw), perc_lid_raw, 0.0)

    colors_lid = [BAR_RED if p < highlight_threshold else BAR_GREEN for p in perc_lid]

    fig, ax = plt.subplots(figsize=figsize)
    y_pos = np.arange(len(possiveis.index))

    max_val = poss_vals.max() if len(possiveis) else 1.0
    if max_val <= 0:
        max_val = 1.0

    ax.barh(y_pos, poss_vals, color=BAR_GRAY, edgecolor="none")
    ax.barh(y_pos, dds_vals, color=BAR_DARK_GRAY, edgecolor="none")
    ax.barh(y_pos, lid_vals, color=colors_lid, edgecolor="none")

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

    for i, (s, p) in enumerate(zip(lid_vals, poss_vals)):
        label = f"{int(s)}/{int(p)}"
        ax.text(s + desloc, i, label, va="center", ha="left", fontsize=10)

    plt.tight_layout()
    return fig


def plot_supervisor_potencial_real(
    possiveis: pd.Series,
    dds_realizados: pd.Series,
    dias_com_lideranca: pd.Series,
    title: str,
    figsize=(12, 5),
    highlight_threshold: float = 60.0,
):
    """
    Idêntico ao anterior, mas o rótulo mostra apenas o percentual
    da liderança em relação aos dias possíveis.
    """
    dds_realizados = dds_realizados.reindex(possiveis.index).fillna(0)
    dias_com_lideranca = dias_com_lideranca.reindex(possiveis.index).fillna(0)

    poss_vals = np.where(
        np.isfinite(pd.to_numeric(possiveis.values, errors="coerce")),
        pd.to_numeric(possiveis.values, errors="coerce"),
        0.0,
    )
    dds_vals = np.where(
        np.isfinite(pd.to_numeric(dds_realizados.values, errors="coerce")),
        pd.to_numeric(dds_realizados.values, errors="coerce"),
        0.0,
    )
    lid_vals = np.where(
        np.isfinite(pd.to_numeric(dias_com_lideranca.values, errors="coerce")),
        pd.to_numeric(dias_com_lideranca.values, errors="coerce"),
        0.0,
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        perc_lid_raw = np.where(poss_vals > 0, (lid_vals / poss_vals) * 100, 0.0)
    perc_lid = np.where(np.isfinite(perc_lid_raw), perc_lid_raw, 0.0)

    colors_lid = [BAR_RED if p < highlight_threshold else BAR_GREEN for p in perc_lid]

    fig, ax = plt.subplots(figsize=figsize)
    y_pos = np.arange(len(possiveis.index))

    max_val = poss_vals.max() if len(possiveis) else 1.0
    if max_val <= 0:
        max_val = 1.0

    ax.barh(y_pos, poss_vals, color=BAR_GRAY, edgecolor="none")
    ax.barh(y_pos, dds_vals, color=BAR_DARK_GRAY, edgecolor="none")
    ax.barh(y_pos, lid_vals, color=colors_lid, edgecolor="none")

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

    for i, (s, perc) in enumerate(zip(lid_vals, perc_lid)):
        label = f"{perc:.1f} %"
        ax.text(s + desloc, i, label, va="center", ha="left", fontsize=10)

    plt.tight_layout()
    return fig


def figure_to_png_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()


# -------------------------------
# Preparação dos dados de DDS
# -------------------------------
def preparar_dds_basico(df: pd.DataFrame, data_ini: date, data_fim: date):
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

    df[COL_HORA_CONCLUSAO] = pd.to_datetime(df[COL_HORA_CONCLUSAO])
    df["data"] = df[COL_HORA_CONCLUSAO].dt.date

    dds = df[df[COL_FORMULARIO] == VALOR_DDS].copy()
    if dds.empty:
        raise ValueError("Não existem registros de DDS na planilha.")

    data_ini = pd.to_datetime(data_ini).date()
    data_fim = pd.to_datetime(data_fim).date()

    mascara = (dds["data"] >= data_ini) & (dds["data"] <= data_fim)
    dds_periodo = dds.loc[mascara].copy()
    if dds_periodo.empty:
        raise ValueError("Não existem registros de DDS no período selecionado.")

    dds_periodo = dds_periodo.sort_values(COL_HORA_CONCLUSAO)
    dds_unico = dds_periodo.drop_duplicates(
        subset=["data", COL_UNIDADE], keep="last"
    ).copy()

    dds_unico["regional_presente"] = dds_unico[COL_PRES_REGIONAL].eq("SIM")
    dds_unico["supervisor_presente"] = dds_unico[COL_PRES_SUPERVISOR].eq("SIM")

    data_ini_efetiva = dds_unico["data"].min()
    data_fim_efetiva = dds_unico["data"].max()
    return dds_unico, data_ini_efetiva, data_fim_efetiva


# -------------------------------
# Análise e geração de gráficos
# -------------------------------
def analisar_dds(
    df: pd.DataFrame,
    data_ini: date,
    data_fim: date,
    possiveis_dds_por_unidade: pd.Series,
    df_dias_regional: pd.DataFrame,
    dias_possiveis_gt: int,
    dias_participacao_gt: int,
):
    # Base única de DDS no período
    dds_unico, data_ini_efetiva, data_fim_efetiva = preparar_dds_basico(
        df, data_ini, data_fim
    )

    reg_text = dds_unico[COL_REGIONAL].astype(str)
    reg_norm = reg_text.map(normalizar_texto)

    is_r1 = reg_norm.eq("R1")
    is_r2 = reg_norm.eq("R2")
    is_r3 = reg_norm.eq("R3")
    is_logarte_reg = reg_text.map(eh_logarte_ou_artepecas)
    is_csc = reg_norm.str.contains("CSC", na=False)
    is_comercial = dds_unico[COL_UNIDADE].eq("COMERCIAL")

    # Base sem COMERCIAL (para quase todos os gráficos)
    dds_sem_comercial = dds_unico[~is_comercial].copy()
    idx_sem = dds_sem_comercial.index

    reg_norm_sem = reg_norm.loc[idx_sem]
    is_csc_sem = is_csc.loc[idx_sem]
    is_r3_sem = is_r3.loc[idx_sem]
    is_logarte_sem = is_logarte_reg.loc[idx_sem]
    is_r1_sem = is_r1.loc[idx_sem]
    is_r2_sem = is_r2.loc[idx_sem]

    # Flags de presença linha a linha (base sem COMERCIAL)

    # Supervisor presente (independente da regional) – CSC fora
    sup_flag_sem = (
        dds_sem_comercial["supervisor_presente"]
        & (~is_csc_sem)
    )

    # LogArte / ArtePeças vindo em REGIONAL conta como liderança da unidade
    reg_logarte_flag_sem = (
        dds_sem_comercial["regional_presente"]
        & is_logarte_sem
        & (~is_csc_sem)
        & (~is_r3_sem)
    )

    # Regionais R1/R2 presentes na unidade (não CSC, não R3, não LogArte)
    reg_r1r2_flag_unit_sem = (
        dds_sem_comercial["regional_presente"]
        & (is_r1_sem | is_r2_sem)
        & (~is_csc_sem)
        & (~is_r3_sem)
        & (~is_logarte_sem)
    )

    # Liderança que conta para a unidade:
    # supervisor + LogArte/ArtePeças + R1/R2
    sup_ou_logarte_flag_sem = (
        sup_flag_sem
        | reg_logarte_flag_sem
        | reg_r1r2_flag_unit_sem
    )

    # Regionais "válidos" para o gráfico de participação de regional por regional:
    # sem R3, sem CSC, sem LogArte
    reg_valido_flag_sem = (
        dds_sem_comercial["regional_presente"]
        & (~is_csc_sem)
        & (~is_r3_sem)
        & (~is_logarte_sem)
    )

    figs: Dict[str, plt.Figure] = {}

    # DDS realizados por unidade
    dds_realizados_por_unidade_all = dds_unico.groupby(COL_UNIDADE).size()
    dds_realizados_por_unidade_sem = dds_sem_comercial.groupby(COL_UNIDADE).size()

    # 5) Participação dos regionais – sem R3 / CSC / LogArte
    df_regional_presente = dds_sem_comercial[reg_valido_flag_sem].copy()
    if not df_regional_presente.empty:
        freq_regional_por_regional = (
            df_regional_presente.groupby(COL_REGIONAL)
            .size()
            .sort_values(ascending=False)
        )
        figs["participacao_regional_por_regional"] = plot_bar_with_labels(
            freq_regional_por_regional,
            title="Participação dos regionais",
            xlabel="",
            ylabel="",
            rotation=0,
            figsize=(8, 4),
            horizontal=False,
        )

    # 2 e 3) Liderança por unidade + R1/R2 + GT (gráficos de participação vs dias possíveis)

    # Unidades (sem COMERCIAL)
    poss_unit = possiveis_dds_por_unidade.reindex(
        dds_realizados_por_unidade_sem.index
    ).astype(float).fillna(0.0)

    dds_unit = dds_realizados_por_unidade_sem.reindex(poss_unit.index).astype(float).fillna(0.0)

    # Dias com liderança por unidade (sup + R1/R2 + LogArte)
    sup_dds_por_unidade = (
        dds_sem_comercial[sup_ou_logarte_flag_sem]
        .drop_duplicates(["data", COL_UNIDADE])
        .groupby(COL_UNIDADE)
        .size()
        if sup_ou_logarte_flag_sem.any()
        else pd.Series(dtype=float)
    )
    lider_unit = sup_dds_por_unidade.reindex(poss_unit.index).astype(float).fillna(0.0)

    # Por regional (R1, R2)

    # Dias com DDS por regional (independente de presença do regional)
    base_dds_reg = dds_sem_comercial[~is_csc_sem].dropna(subset=[COL_REGIONAL]).copy()
    dds_por_regional_dias = (
        base_dds_reg
        .drop_duplicates(["data", COL_REGIONAL])
        .groupby(COL_REGIONAL)
        .size()
    )

    # Dias em que R1/R2 estiveram presentes em algum DDS
    base_pres_reg = dds_sem_comercial[
        dds_sem_comercial["regional_presente"]
        & (~is_csc_sem)
        & (~is_r3_sem)
        & (~is_logarte_sem)
        & (is_r1_sem | is_r2_sem)
    ].drop_duplicates(["data", COL_REGIONAL])

    pres_por_regional_dias = base_pres_reg.groupby(COL_REGIONAL).size()

    reg_labels = ["R1", "R2"]

    if not df_dias_regional.empty and "Regional" in df_dias_regional.columns:
        poss_reg = (
            df_dias_regional.set_index("Regional")["Dias possíveis"]
            .astype(float)
            .reindex(reg_labels)
        )
    else:
        poss_reg = pd.Series(0.0, index=reg_labels)

    poss_reg = poss_reg.fillna(0.0)
    dds_reg = dds_por_regional_dias.reindex(reg_labels).astype(float).fillna(0.0)
    pres_reg = pres_por_regional_dias.reindex(reg_labels).astype(float).fillna(0.0)

    # Gerente de Tecnologia (GT) - totalmente manual
    poss_gt = float(dias_possiveis_gt)
    pres_gt = float(dias_participacao_gt)
    pres_gt = max(0.0, min(pres_gt, poss_gt))

    poss_gt_series = pd.Series([poss_gt], index=["TECNOLOGIA DO CONCRETO"])
    # Para GT, consideramos que todos os dias possíveis são dias "com DDS" no nível agregado
    dds_gt_series = pd.Series([poss_gt], index=["TECNOLOGIA DO CONCRETO"])
    # E os dias de participação vêm apenas do valor digitado
    pres_gt_series = pd.Series([pres_gt], index=["TECNOLOGIA DO CONCRETO"])

    # Combina unidades + regionais (R1/R2) + GT
    poss_comb = pd.concat([poss_unit, poss_reg, poss_gt_series])
    dds_comb = pd.concat([dds_unit, dds_reg, dds_gt_series])
    lider_comb = pd.concat([lider_unit, pres_reg, pres_gt_series])

    # remove linhas sem dias possíveis
    mask_valid = poss_comb.fillna(0) > 0
    poss_comb = poss_comb[mask_valid]
    dds_comb = dds_comb[mask_valid]
    lider_comb = lider_comb[mask_valid]

    # ordena pela quantidade de dias com liderança
    lider_comb = lider_comb.sort_values(ascending=False)
    poss_comb = poss_comb.reindex(lider_comb.index)
    dds_comb = dds_comb.reindex(lider_comb.index)

    if not lider_comb.empty:
        # 2) Participação da liderança x dias possíveis
        figs["participacao_supervisor_por_unidade_contagem"] = plot_supervisor_potencial_real_counts(
            poss_comb,
            dds_comb,
            lider_comb,
            title="Participação da liderança x dias possíveis",
            figsize=(10, max(5, len(poss_comb) * 0.3)),
            highlight_threshold=60.0,
        )

        # 3) Participação da liderança x dias possíveis (%)
        figs["participacao_supervisor_por_unidade_percentual_camadas"] = plot_supervisor_potencial_real(
            poss_comb,
            dds_comb,
            lider_comb,
            title="Participação da liderança x dias possíveis (%)",
            figsize=(10, max(5, len(poss_comb) * 0.3)),
            highlight_threshold=60.0,
        )

    # 4) Percentual de presença dos supervisores por regional (R1, R2, R3)
    base_regional = dds_sem_comercial.copy()
    reg_norm_sem_full = reg_norm.loc[base_regional.index]

    mask_r123 = reg_norm_sem_full.isin(["R1", "R2", "R3"])
    base_r123 = base_regional[mask_r123]

    total_dds_por_regional = (
        base_r123.groupby(COL_REGIONAL).size()
        if not base_r123.empty
        else pd.Series(dtype=int)
    )

    sup_mask_r123 = sup_flag_sem.reindex(base_regional.index, fill_value=False) & mask_r123
    sup_r123 = base_regional[sup_mask_r123]

    sup_dds_por_regional = (
        sup_r123.groupby(COL_REGIONAL).size()
        if not sup_r123.empty
        else pd.Series(dtype=int)
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
            title="Presença de supervisores por regional (%)",
            xlabel="",
            ylabel="",
            rotation=0,
            figsize=(8, 4),
            horizontal=False,
            is_percent=True,
        )

    # 1) DDS realizados x dias possíveis (unidade / regional / GT)
    #    - inclui COMERCIAL
    #    - remove CSC (no nome da unidade)
    #    - regionais: R1 / R2
    #    - GT com barra própria

    unidades_index = dds_realizados_por_unidade_all.index.to_series()
    mask_csc_unidade = unidades_index.astype(str).map(normalizar_texto).str.contains(
        "CSC", na=False
    )
    dds_unid_sem_csc = dds_realizados_por_unidade_all[~mask_csc_unidade].astype(float)

    poss_unid_sem_csc = possiveis_dds_por_unidade.reindex(dds_unid_sem_csc.index).astype(float)

    # Regionais R1 / R2 – dias com DDS
    dds_reg_dias_all = (
        dds_unico[reg_norm.isin(["R1", "R2"])]
        .drop_duplicates(["data", COL_REGIONAL])
        .groupby(COL_REGIONAL)
        .size()
    )

    if not df_dias_regional.empty and "Regional" in df_dias_regional.columns:
        poss_reg_all = (
            df_dias_regional.set_index("Regional")["Dias possíveis"]
            .astype(float)
            .reindex(["R1", "R2"])
        )
    else:
        poss_reg_all = pd.Series(0.0, index=["R1", "R2"])

    poss_reg_all = poss_reg_all.fillna(0.0)
    dds_reg_all = dds_reg_dias_all.reindex(["R1", "R2"]).astype(float).fillna(0.0)

    # GT
    poss_gt_all = poss_gt_series
    dds_gt_all = pd.Series([pres_gt], index=["TECNOLOGIA DO CONCRETO"])

    poss_comb_all = pd.concat([poss_unid_sem_csc, poss_reg_all, poss_gt_all])
    dds_comb_all = pd.concat([dds_unid_sem_csc, dds_reg_all, dds_gt_all])

    mask_valid2 = poss_comb_all.fillna(0) > 0
    poss_comb_all = poss_comb_all[mask_valid2]
    dds_comb_all = dds_comb_all[mask_valid2]

    with np.errstate(divide="ignore", invalid="ignore"):
        perc_real_all = dds_comb_all / poss_comb_all
    perc_real_all = perc_real_all.replace([np.inf, -np.inf], 0).fillna(0)

    idx_sorted = perc_real_all.sort_values(ascending=False).index

    poss_sorted_all = poss_comb_all.reindex(idx_sorted)
    dds_sorted_all = dds_comb_all.reindex(idx_sorted)

    figs["dias_possiveis_x_realizados_por_unidade"] = plot_dds_potencial_real(
        poss_sorted_all,
        dds_sorted_all,
        title="DDS realizados x dias possíveis",
        figsize=(10, max(5, len(poss_sorted_all) * 0.3)),
        highlight_threshold=60.0,
    )

    return dds_unico, figs, data_ini_efetiva, data_fim_efetiva


# -------------------------------
# STREAMLIT APP
# -------------------------------
def main():
    st.set_page_config(page_title="Análise de DDS - Concrearte", layout="wide")
    st.title("📊 Análise de DDS - Supervisores, Regionais e Tecnologia do Concreto")

    st.markdown(
        """
Este app lê a planilha de **SSMA**, filtra os **DDS** e gera gráficos de:

1. **DDS realizados x dias possíveis**
2. **Participação da liderança x dias possíveis**
3. **Participação da liderança x dias possíveis (%)**
4. **Presença de supervisores por regional (%)**
5. **Participação dos regionais**
"""
    )

    # Sidebar: upload + período
    st.sidebar.header("⚙️ Arquivo e período")

    uploaded_file = st.sidebar.file_uploader(
        "1) Envie a planilha de SSMA (.xlsx)", type=["xlsx", "xls"]
    )

    if uploaded_file is None:
        st.info("Envie a planilha na barra lateral para começar.")
        return

    try:
        df_raw = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Erro ao ler a planilha: {e}")
        return

    if COL_HORA_CONCLUSAO not in df_raw.columns:
        st.error(f"A coluna '{COL_HORA_CONCLUSAO}' não foi encontrada na planilha.")
        return

    try:
        datas = pd.to_datetime(df_raw[COL_HORA_CONCLUSAO])
        data_min = datas.min().date()
        data_max = datas.max().date()
    except Exception:
        st.error("Não foi possível converter a coluna de hora de conclusão para data.")
        return

    st.sidebar.markdown("### 2) Período de análise")
    data_ini_input = st.sidebar.date_input(
        "Data inicial",
        value=data_min,
        min_value=data_min,
        max_value=data_max,
    )
    data_fim_input = st.sidebar.date_input(
        "Data final",
        value=data_max,
        min_value=data_min,
        max_value=data_max,
    )

    if data_ini_input > data_fim_input:
        st.sidebar.error("A data inicial não pode ser maior que a data final.")
        return

    # Prévia DDS + unidades (para montar tabela de unidades)
    try:
        dds_preview, _, _ = preparar_dds_basico(df_raw, data_ini_input, data_fim_input)
    except ValueError as e:
        st.error(f"Erro ao preparar dados de DDS: {e}")
        return

    st.markdown("## 3) Ajustar dias possíveis por unidade")

    unidades = (
        dds_preview[COL_UNIDADE]
        .dropna()
        .astype(str)
        .sort_values()
        .unique()
    )

    dias_todos = pd.date_range(data_ini_input, data_fim_input, freq="D")

    lista_unidades = []
    for un in unidades:
        if unidade_e_logarte_ou_artepecas(un):
            dias_pos = dias_todos[dias_todos.weekday < 5].size  # seg–sex
        else:
            dias_pos = dias_todos[dias_todos.weekday < 6].size  # seg–sáb
        lista_unidades.append({"Unidade": un, "Dias possíveis": int(dias_pos)})

    df_unid_default = pd.DataFrame(lista_unidades)

    # Remove eventual unidade "CSC"
    mask_csc_unid = df_unid_default["Unidade"].astype(str).map(normalizar_texto).str.contains(
        "CSC", na=False
    )
    df_unid_default = df_unid_default[~mask_csc_unid].reset_index(drop=True)

    st.caption(
        "Ajuste os **dias possíveis** por unidade (LogArte/ArtePeças já vêm como seg–sex; "
        "demais unidades como seg–sáb)."
    )
    df_unid_edit = st.data_editor(
        df_unid_default,
        num_rows="fixed",
        use_container_width=True,
        key="tbl_unidades",
    )

    # Dias possíveis para regionais (R1 / R2 / R3)
    st.markdown("## 4) Ajustar dias possíveis para Regionais (R1 / R2 / R3)")

    df_regional_default = pd.DataFrame(
        [
            {"Regional": "R1", "Dias possíveis": 0},
            {"Regional": "R2", "Dias possíveis": 0},
            {"Regional": "R3", "Dias possíveis": 0},
        ]
    )

    st.caption(
        "Informe os **dias possíveis** em que cada regional poderia ter acompanhado DDS no período."
    )
    df_regional_edit = st.data_editor(
        df_regional_default,
        num_rows="fixed",
        use_container_width=True,
        key="tbl_regionais",
    )

    # Parâmetros do Gerente de Tecnologia
    st.markdown("## 5) Parâmetros do Gerente de Tecnologia")

    col1, col2 = st.columns(2)
    with col1:
        dias_possiveis_gt = st.number_input(
            "Dias possíveis para o Gerente de Tecnologia",
            min_value=0,
            step=1,
            value=0,
        )
    with col2:
        dias_participacao_gt = st.number_input(
            "Dias em que o Gerente de Tecnologia participou",
            min_value=0,
            step=1,
            value=0,
        )

    # Botão de gerar
    st.markdown("## 6) Gerar gráficos")
    gerar = st.button("Gerar gráficos", type="primary")

    if not gerar:
        return

    possiveis_dds_por_unidade = df_unid_edit.set_index("Unidade")["Dias possíveis"].astype(float)
    df_dias_regional = df_regional_edit.copy()

    try:
        dds_unico, figs, data_ini_efetiva, data_fim_efetiva = analisar_dds(
            df_raw,
            data_ini=data_ini_input,
            data_fim=data_fim_input,
            possiveis_dds_por_unidade=possiveis_dds_por_unidade,
            df_dias_regional=df_dias_regional,
            dias_possiveis_gt=int(dias_possiveis_gt),
            dias_participacao_gt=int(dias_participacao_gt),
        )
    except ValueError as e:
        st.error(f"Erro na análise: {e}")
        return

    st.success(
        f"Análise concluída! Período considerado: **{data_ini_efetiva}** a "
        f"**{data_fim_efetiva}**. "
        f"Total de registros únicos (1 por dia/unidade): **{len(dds_unico)}**."
    )

    st.subheader("📈 Gráficos")

    # Ordem desejada dos gráficos
    ordem_graficos = [
        "dias_possiveis_x_realizados_por_unidade",          # 1
        "participacao_supervisor_por_unidade_contagem",     # 2
        "participacao_supervisor_por_unidade_percentual_camadas",  # 3
        "percentual_supervisor_por_regional",               # 4
        "participacao_regional_por_regional",               # 5
    ]

    for nome in ordem_graficos:
        if nome not in figs:
            continue
        fig = figs[nome]
        titulo = fig.axes[0].get_title()
        st.markdown(f"### {titulo}")
        st.pyplot(fig)

        png_bytes = figure_to_png_bytes(fig)
        st.download_button(
            label="⬇️ Baixar este gráfico em PNG",
            data=png_bytes,
            file_name=f"{nome}.png",
            mime="image/png",
            key=f"download_{nome}",
        )
        plt.close(fig)


if __name__ == "__main__":
    main()
