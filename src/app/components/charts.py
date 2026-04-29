import numpy as np
import pandas as pd
import plotly.graph_objects as go

TEAM_COLORS = {
    "KIA":  "#ea0029",
    "삼성":  "#074CA1",
    "LG":   "#a50034",
    "두산":  "#1a1748",
    "KT":   "#333333",
    "SSG":  "#ce0e2d",
    "롯데":  "#041E42",
    "한화":  "#FC4E00",
    "NC":   "#315288",
    "키움":  "#570514",
}

BG          = "#FFFFFF"
CUTLINE_CLR = "#E53935"
_FONT       = dict(family="'Noto Sans KR', -apple-system, sans-serif", size=12)
_HOVER      = dict(bgcolor="white", bordercolor="#E2E8F0",
                   font=dict(size=13, family="'Noto Sans KR', -apple-system, sans-serif"))


def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _base_layout(**kwargs) -> dict:
    base = dict(
        plot_bgcolor=BG, paper_bgcolor=BG,
        font=_FONT,
        hoverlabel=_HOVER,
        margin=dict(l=10, r=20, t=24, b=44),
    )
    base.update(kwargs)
    return base


def bar_chart(latest: pd.DataFrame, top5_teams: set) -> go.Figure:
    df = latest.sort_values("prob_norm", ascending=True).reset_index(drop=True)

    bar_colors  = []
    text_colors = []
    for team in df["team"]:
        c = TEAM_COLORS.get(team, "#888")
        if team in top5_teams:
            bar_colors.append(c)
            text_colors.append(c)
        else:
            bar_colors.append(_rgba(c, 0.22))
            text_colors.append("#A0AEC0")

    fig = go.Figure()

    # 상위 5팀 배경 하이라이트
    fig.add_hrect(
        y0=4.5, y1=9.5,
        fillcolor="rgba(239,246,255,0.6)", line_width=0,
        layer="below",
    )

    fig.add_trace(go.Bar(
        x=df["prob_norm"],
        y=df["team"],
        orientation="h",
        marker=dict(
            color=bar_colors,
            line=dict(color="white", width=0.5),
        ),
        text=[f"<b>{v:.1%}</b>" if t in top5_teams else f"{v:.1%}"
              for v, t in zip(df["prob_norm"], df["team"])],
        textposition="outside",
        textfont=dict(size=13),
        customdata=df[["wins", "losses", "win_rate"]].values,
        hovertemplate=(
            "<b>%{y}</b><br>"
            "진출 확률: <b>%{x:.1%}</b><br>"
            "%{customdata[0]}승 %{customdata[1]}패 (승률 %{customdata[2]:.3f})"
            "<extra></extra>"
        ),
    ))

    # 컷라인
    fig.add_hline(
        y=4.5, line_dash="dash", line_color=CUTLINE_CLR, line_width=1.6, opacity=0.75,
        annotation_text="포스트시즌 컷라인",
        annotation_font=dict(color=CUTLINE_CLR, size=11),
        annotation_position="top right",
    )
    fig.add_vline(x=0.5, line_dash="dot", line_color="#BBBBBB", line_width=1, opacity=0.7)

    fig.update_layout(
        **_base_layout(height=420),
        xaxis=dict(
            tickformat=".0%", range=[0, 1.18],
            showgrid=True, gridcolor="#F0F0F0", zeroline=False,
            title=dict(text="포스트시즌 진출 확률", font=dict(size=11, color="#64748B")),
        ),
        yaxis=dict(showgrid=False, tickfont=dict(size=12)),
        showlegend=False,
    )
    return fig


def trend_chart(pred_df: pd.DataFrame, top5_teams: set) -> go.Figure:
    fig = go.Figure()

    fig.add_hrect(y0=0.5, y1=1.05, fillcolor="rgba(27,63,122,0.03)", line_width=0)
    fig.add_hline(
        y=0.5, line_dash="dash", line_color=CUTLINE_CLR, line_width=1.3, opacity=0.65,
        annotation_text="포스트시즌 기준선",
        annotation_font=dict(color=CUTLINE_CLR, size=10),
        annotation_position="top left",
    )

    # 하위팀 먼저(뒤에 렌더)
    for team in sorted(pred_df["team"].unique()):
        if team in top5_teams:
            continue
        t = pred_df[pred_df["team"] == team].sort_values("games")
        c = TEAM_COLORS.get(team, "#888")
        fig.add_trace(go.Scatter(
            x=t["games"], y=t["prob_norm"],
            mode="lines", name=team, showlegend=False,
            line=dict(color=_rgba(c, 0.35), width=1.0, dash="dot"),
            hovertemplate=f"<b>{team}</b><br>경기수: %{{x}}<br>확률: %{{y:.1%}}<extra></extra>",
        ))

    # 상위 5팀 — fill 포함
    for team in sorted(pred_df["team"].unique()):
        if team not in top5_teams:
            continue
        t = pred_df[pred_df["team"] == team].sort_values("games")
        c = TEAM_COLORS.get(team, "#888")
        fig.add_trace(go.Scatter(
            x=t["games"], y=t["prob_norm"],
            mode="lines",
            name=team,
            line=dict(color=c, width=2.6),
            hovertemplate=f"<b>{team}</b><br>경기수: %{{x}}<br>확률: %{{y:.1%}}<extra></extra>",
        ))

    fig.update_layout(
        **_base_layout(height=460, margin=dict(l=10, r=10, t=50, b=44)),
        xaxis=dict(
            title=dict(text="누적 경기 수", font=dict(size=11, color="#64748B")),
            showgrid=True, gridcolor="#F0F0F0", zeroline=False,
        ),
        yaxis=dict(
            title=dict(text="포스트시즌 진출 확률", font=dict(size=11, color="#64748B")),
            tickformat=".0%", range=[0, 1.08],
            showgrid=True, gridcolor="#F0F0F0",
        ),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.01, xanchor="center", x=0.5,
            font=dict(size=11), bgcolor="rgba(255,255,255,0.8)",
            bordercolor="#E2E8F0", borderwidth=1,
        ),
    )
    return fig


def bump_chart(rank_df: pd.DataFrame, top5_teams: set) -> go.Figure:
    fig = go.Figure()

    dates_ordered = sorted(rank_df["date"].unique())
    d2x = {d: i for i, d in enumerate(dates_ordered)}
    n   = len(dates_ordered)
    tick_step  = max(1, n // 8)
    tick_vals  = list(range(0, n, tick_step))
    tick_text  = [pd.to_datetime(dates_ordered[i]).strftime("%m/%d") for i in tick_vals]

    # 포스트시즌 배경
    fig.add_hrect(y0=0.5, y1=5.5, fillcolor="rgba(27,63,122,0.03)", line_width=0)
    fig.add_hline(
        y=5.5, line_dash="dot", line_color=CUTLINE_CLR, line_width=1.3, opacity=0.7,
        annotation_text="포스트시즌 컷라인",
        annotation_font=dict(color=CUTLINE_CLR, size=10),
        annotation_position="bottom left",
    )

    for team in sorted(rank_df["team"].unique()):
        t      = rank_df[rank_df["team"] == team].sort_values("date")
        xs     = [d2x[d] for d in t["date"]]
        ys     = t["pred_rank"].values
        is_top = team in top5_teams
        c      = TEAM_COLORS.get(team, "#888")

        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            mode="lines+markers" if is_top else "lines",
            name=team,
            line=dict(
                color=c if is_top else _rgba(c, 0.30),
                width=2.6 if is_top else 1.0,
                dash="solid" if is_top else "dot",
            ),
            marker=dict(size=5 if is_top else 0, color=c,
                        line=dict(color="white", width=1.5)),
            opacity=1.0 if is_top else 1.0,
            showlegend=is_top,
            hovertemplate=f"<b>{team}</b><br>%{{customdata}}<br>순위: %{{y}}위<extra></extra>",
            customdata=[pd.to_datetime(d).strftime("%m/%d") for d in t["date"]],
        ))

    fig.update_layout(
        **_base_layout(height=460, margin=dict(l=10, r=10, t=50, b=44)),
        xaxis=dict(
            tickvals=tick_vals, ticktext=tick_text,
            showgrid=True, gridcolor="#F0F0F0", zeroline=False,
            title=dict(text="날짜", font=dict(size=11, color="#64748B")),
        ),
        yaxis=dict(
            title=dict(text="예측 순위", font=dict(size=11, color="#64748B")),
            tickvals=list(range(1, 11)),
            ticktext=[f"{i}위" for i in range(1, 11)],
            autorange="reversed",
            showgrid=True, gridcolor="#F0F0F0",
        ),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.01, xanchor="center", x=0.5,
            font=dict(size=11), bgcolor="rgba(255,255,255,0.8)",
            bordercolor="#E2E8F0", borderwidth=1,
        ),
    )
    return fig


def scatter_chart(latest: pd.DataFrame, top5_teams: set) -> go.Figure:
    fig = go.Figure()

    # 사분면 배경
    q_kw = dict(layer="below", line_width=0)
    fig.add_hrect(y0=0.5, y1=1.1,  fillcolor="rgba(219,234,254,0.25)", **q_kw)
    fig.add_hrect(y0=0.0, y1=0.5,  fillcolor="rgba(241,245,249,0.30)", **q_kw)
    fig.add_vrect(x0=0.5, x1=0.82, fillcolor="rgba(254,240,138,0.12)", **q_kw)

    # 대각선
    diag = np.linspace(0.20, 0.82, 100)
    fig.add_trace(go.Scatter(
        x=diag, y=diag, mode="lines",
        line=dict(color="#CCCCCC", dash="dash", width=1.2),
        showlegend=False, hoverinfo="skip",
    ))

    # 기준선
    fig.add_vline(x=0.5, line_color="#DDDDDD", line_width=1)
    fig.add_hline(y=0.5, line_color="#DDDDDD", line_width=1)

    # 사분면 레이블
    lbl_kw = dict(font=dict(size=9, color="#B0BEC5"), showarrow=False)
    fig.add_annotation(x=0.37, y=0.80, text="현재 부진<br>모델 낙관", **lbl_kw)
    fig.add_annotation(x=0.70, y=0.80, text="현재 강세<br>모델 낙관", **lbl_kw)
    fig.add_annotation(x=0.37, y=0.20, text="현재 부진<br>모델 비관", **lbl_kw)
    fig.add_annotation(x=0.70, y=0.20, text="현재 강세<br>모델 비관", **lbl_kw)

    for _, row in latest.iterrows():
        team   = row["team"]
        is_top = team in top5_teams
        c      = TEAM_COLORS.get(team, "#888")

        fig.add_trace(go.Scatter(
            x=[row["win_rate"]], y=[row["prob_norm"]],
            mode="markers+text", name=team, showlegend=False,
            marker=dict(
                size=18 if is_top else 11,
                color=c if is_top else _rgba(c, 0.38),
                line=dict(color="white", width=2),
                symbol="circle",
            ),
            text=[f"<b>{team}</b>" if is_top else team],
            textposition="top right",
            textfont=dict(size=11 if is_top else 9,
                          color=c if is_top else _rgba(c, 0.6)),
            hovertemplate=(
                f"<b>{team}</b><br>"
                "현재 승률: %{x:.1%}<br>"
                "모델 예측: %{y:.1%}<extra></extra>"
            ),
        ))

    fig.update_layout(
        **_base_layout(height=440),
        xaxis=dict(
            title=dict(text="현재 승률", font=dict(size=11, color="#64748B")),
            tickformat=".0%", range=[0.20, 0.84],
            showgrid=True, gridcolor="#F0F0F0",
        ),
        yaxis=dict(
            title=dict(text="모델 예측 확률", font=dict(size=11, color="#64748B")),
            tickformat=".0%", range=[0.0, 1.10],
            showgrid=True, gridcolor="#F0F0F0",
        ),
        showlegend=False,
    )
    return fig


def importance_chart(importance: pd.Series) -> go.Figure:
    top20 = importance.sort_values(ascending=False).head(20)

    def _group(name):
        if name.startswith("dyn_"):  return ("#2E8B57", "3년 평균 역가중")
        if name.startswith("prev_"): return ("#2563A8", "전년도 기록")
        return ("#94A3B8", "현재 시즌")

    colors = [_group(f)[0] for f in top20.index[::-1]]
    labels = list(top20.index[::-1])
    values = list(top20.values[::-1])

    fig = go.Figure(go.Bar(
        x=values, y=labels,
        orientation="h",
        marker=dict(color=colors, line=dict(color="white", width=0.4)),
        text=[f"{v:.4f}" for v in values],
        textposition="outside",
        textfont=dict(size=10, color="#475569"),
        hovertemplate="<b>%{y}</b><br>중요도: %{x:.4f}<extra></extra>",
    ))

    fig.update_layout(
        **_base_layout(height=530, margin=dict(l=10, r=70, t=24, b=44)),
        xaxis=dict(showgrid=True, gridcolor="#F0F0F0", zeroline=False,
                   title=dict(text="중요도 (3모델 평균 정규화)", font=dict(size=11, color="#64748B"))),
        yaxis=dict(showgrid=False, tickfont=dict(size=10)),
        showlegend=False,
    )
    return fig


def heatmap_chart(pred_df: pd.DataFrame, top5_teams: set) -> go.Figure:
    latest     = pred_df.sort_values("date").groupby("team").last().reset_index()
    team_order = list(latest.sort_values("prob_norm", ascending=False)["team"])
    dates_list = sorted(pred_df["date"].unique())
    dlabels    = [pd.to_datetime(d).strftime("%m/%d") for d in dates_list]

    z = np.full((len(team_order), len(dates_list)), np.nan)
    for i, team in enumerate(team_order):
        sub = pred_df[pred_df["team"] == team].set_index("date")["prob_norm"]
        for j, date in enumerate(dates_list):
            if date in sub.index:
                z[i, j] = sub[date]

    y_labels = [f"{'★  ' if t in top5_teams else '     '}{t}" for t in team_order]
    text_z   = [[f"{v:.0%}" if not np.isnan(v) else "" for v in row] for row in z]

    custom_scale = [
        [0.0,  "#EFF6FF"], [0.25, "#BFDBFE"],
        [0.50, "#3B82F6"], [0.75, "#1D4ED8"],
        [1.0,  "#1E3A8A"],
    ]

    fig = go.Figure(go.Heatmap(
        z=z, x=dlabels, y=y_labels,
        text=text_z, texttemplate="%{text}",
        textfont=dict(size=9),
        colorscale=custom_scale,
        zmin=0, zmax=1,
        hovertemplate="<b>%{y}</b><br>날짜: %{x}<br>확률: %{z:.1%}<extra></extra>",
        colorbar=dict(
            title=dict(text="진출 확률", font=dict(size=10, color="#64748B")),
            tickformat=".0%", len=0.85,
            outlinewidth=0, bgcolor="white",
        ),
    ))

    fig.add_hline(y=4.5, line_dash="dash", line_color=CUTLINE_CLR,
                  line_width=1.8, opacity=0.85)

    tick_step = max(1, len(dates_list) // 8)
    fig.update_xaxes(tickvals=dlabels[::tick_step], ticktext=dlabels[::tick_step], tickangle=0)
    fig.update_layout(
        **_base_layout(height=390),
        yaxis=dict(tickfont=dict(size=11)),
    )
    return fig
