import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

TEAM_COLORS = {
    "KIA":  "#ea0029",
    "삼성":  "#074CA1",
    "LG":   "#a50034",
    "두산":  "#1a1748",
    "KT":   "#000000",
    "SSG":  "#ce0e2d",
    "롯데":  "#041E42",
    "한화":  "#FC4E00",
    "NC":   "#315288",
    "키움":  "#570514",
}

BG = "#F8F9FA"
CUTLINE_COLOR = "#E53935"


def bar_chart(latest: pd.DataFrame, top5_teams: set) -> go.Figure:
    df = latest.sort_values("prob_norm", ascending=True)

    colors = [
        TEAM_COLORS.get(t, "#888") if t in top5_teams else "#D5D5D5"
        for t in df["team"]
    ]
    text = [f"{v:.1%}" for v in df["prob_norm"]]

    fig = go.Figure(go.Bar(
        x=df["prob_norm"],
        y=df["team"],
        orientation="h",
        marker_color=colors,
        text=text,
        textposition="outside",
        textfont=dict(size=13),
        hovertemplate="<b>%{y}</b><br>진출 확률: %{x:.1%}<extra></extra>",
    ))

    # 컷라인 (5위/6위 경계)
    fig.add_hline(
        y=4.5, line_dash="dash", line_color=CUTLINE_COLOR,
        line_width=1.5, opacity=0.7,
        annotation_text="포스트시즌 컷라인",
        annotation_font_color=CUTLINE_COLOR,
        annotation_position="top right",
    )
    fig.add_vline(x=0.5, line_dash="dot", line_color="#999", line_width=1, opacity=0.6)

    fig.update_layout(
        plot_bgcolor=BG, paper_bgcolor=BG,
        xaxis=dict(
            tickformat=".0%", range=[0, 1.15],
            showgrid=True, gridcolor="#E8E8E8",
            zeroline=False,
        ),
        yaxis=dict(showgrid=False),
        margin=dict(l=10, r=40, t=20, b=40),
        height=420,
        showlegend=False,
    )
    return fig


def trend_chart(pred_df: pd.DataFrame, top5_teams: set) -> go.Figure:
    fig = go.Figure()

    # 포스트시즌 영역
    fig.add_hrect(y0=0.5, y1=1.05, fillcolor="#1B3F7A", opacity=0.04, line_width=0)
    fig.add_hline(y=0.5, line_dash="dash", line_color=CUTLINE_COLOR,
                  line_width=1.2, opacity=0.6,
                  annotation_text="포스트시즌 기준선",
                  annotation_font_color=CUTLINE_COLOR,
                  annotation_position="top left")

    for team in sorted(pred_df["team"].unique()):
        t = pred_df[pred_df["team"] == team].sort_values("games")
        is_top = team in top5_teams
        color  = TEAM_COLORS.get(team, "#888")

        fig.add_trace(go.Scatter(
            x=t["games"], y=t["prob_norm"],
            mode="lines",
            name=team,
            line=dict(
                color=color,
                width=2.4 if is_top else 1.0,
                dash="solid" if is_top else "dash",
            ),
            opacity=1.0 if is_top else 0.35,
            hovertemplate=f"<b>{team}</b><br>경기수: %{{x}}<br>확률: %{{y:.1%}}<extra></extra>",
        ))

    fig.update_layout(
        plot_bgcolor=BG, paper_bgcolor=BG,
        xaxis=dict(title="누적 경기 수", showgrid=True, gridcolor="#E8E8E8", zeroline=False),
        yaxis=dict(title="포스트시즌 진출 확률", tickformat=".0%",
                   range=[0, 1.08], showgrid=True, gridcolor="#E8E8E8"),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.01, xanchor="center", x=0.5,
            font=dict(size=11),
        ),
        margin=dict(l=10, r=10, t=50, b=40),
        height=450,
    )
    return fig


def bump_chart(rank_df: pd.DataFrame, top5_teams: set) -> go.Figure:
    fig = go.Figure()

    dates_ordered = sorted(rank_df["date"].unique())
    date_to_x = {d: i for i, d in enumerate(dates_ordered)}
    tick_step  = max(1, len(dates_ordered) // 8)
    tick_vals  = list(range(0, len(dates_ordered), tick_step))
    tick_text  = [pd.to_datetime(dates_ordered[i]).strftime("%m/%d") for i in tick_vals]

    fig.add_hline(y=5.5, line_dash="dot", line_color=CUTLINE_COLOR,
                  line_width=1.2, opacity=0.7,
                  annotation_text="포스트시즌 컷라인",
                  annotation_font_color=CUTLINE_COLOR,
                  annotation_position="top left")

    for team in sorted(rank_df["team"].unique()):
        t = rank_df[rank_df["team"] == team].sort_values("date")
        xs = [date_to_x[d] for d in t["date"]]
        ys = t["pred_rank"].values
        is_top = team in top5_teams
        color  = TEAM_COLORS.get(team, "#888")

        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            mode="lines+markers" if is_top else "lines",
            name=team,
            line=dict(color=color, width=2.4 if is_top else 1.0,
                      dash="solid" if is_top else "dash"),
            marker=dict(size=4, color=color, line=dict(color="white", width=1)),
            opacity=1.0 if is_top else 0.3,
            hovertemplate=f"<b>{team}</b><br>날짜: %{{x}}<br>순위: %{{y}}위<extra></extra>",
        ))

    fig.update_layout(
        plot_bgcolor=BG, paper_bgcolor=BG,
        xaxis=dict(
            tickvals=tick_vals, ticktext=tick_text,
            showgrid=True, gridcolor="#E8E8E8", zeroline=False,
        ),
        yaxis=dict(
            title="예측 순위",
            tickvals=list(range(1, 11)),
            ticktext=[f"{i}위" for i in range(1, 11)],
            autorange="reversed",
            showgrid=True, gridcolor="#E8E8E8",
        ),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.01, xanchor="center", x=0.5,
            font=dict(size=11),
        ),
        margin=dict(l=10, r=10, t=50, b=40),
        height=450,
    )
    return fig


def scatter_chart(latest: pd.DataFrame, top5_teams: set) -> go.Figure:
    fig = go.Figure()

    diag = np.linspace(0.2, 0.85, 100)
    fig.add_trace(go.Scatter(
        x=diag, y=diag, mode="lines",
        line=dict(color="#CCCCCC", dash="dash", width=1),
        showlegend=False, hoverinfo="skip",
    ))
    fig.add_vline(x=0.5, line_color="#DDDDDD", line_width=1)
    fig.add_hline(y=0.5, line_color="#DDDDDD", line_width=1)

    for _, row in latest.iterrows():
        team   = row["team"]
        is_top = team in top5_teams
        color  = TEAM_COLORS.get(team, "#888")

        fig.add_trace(go.Scatter(
            x=[row["win_rate"]], y=[row["prob_norm"]],
            mode="markers+text",
            name=team,
            marker=dict(
                size=16 if is_top else 10,
                color=color,
                opacity=1.0 if is_top else 0.4,
                line=dict(color="white", width=1.5),
            ),
            text=[team],
            textposition="top right",
            textfont=dict(
                size=11 if is_top else 9,
                color=color,
            ),
            hovertemplate=(
                f"<b>{team}</b><br>"
                "현재 승률: %{x:.1%}<br>"
                "모델 예측: %{y:.1%}<extra></extra>"
            ),
        ))

    fig.update_layout(
        plot_bgcolor=BG, paper_bgcolor=BG,
        xaxis=dict(title="현재 승률", tickformat=".0%",
                   range=[0.22, 0.82], showgrid=True, gridcolor="#E8E8E8"),
        yaxis=dict(title="모델 예측 확률", tickformat=".0%",
                   range=[0, 1.1], showgrid=True, gridcolor="#E8E8E8"),
        showlegend=False,
        margin=dict(l=10, r=10, t=20, b=40),
        height=430,
    )
    return fig


def importance_chart(importance: pd.Series) -> go.Figure:
    top20 = importance.sort_values(ascending=False).head(20)

    def group_color(name):
        if name.startswith("dyn_"):  return "#2E8B57"
        if name.startswith("prev_"): return "#2563A8"
        return "#888888"

    colors = [group_color(f) for f in top20.index[::-1]]
    labels = list(top20.index[::-1])
    values = list(top20.values[::-1])

    fig = go.Figure(go.Bar(
        x=values, y=labels,
        orientation="h",
        marker_color=colors,
        text=[f"{v:.4f}" for v in values],
        textposition="outside",
        textfont=dict(size=10),
        hovertemplate="<b>%{y}</b><br>중요도: %{x:.4f}<extra></extra>",
    ))

    fig.update_layout(
        plot_bgcolor=BG, paper_bgcolor=BG,
        xaxis=dict(showgrid=True, gridcolor="#E8E8E8", zeroline=False),
        yaxis=dict(showgrid=False),
        margin=dict(l=10, r=60, t=20, b=40),
        height=520,
        showlegend=False,
    )
    return fig


def heatmap_chart(pred_df: pd.DataFrame, top5_teams: set) -> go.Figure:
    latest = pred_df.sort_values("date").groupby("team").last().reset_index()
    team_order  = list(latest.sort_values("prob_norm", ascending=False)["team"])
    dates_list  = sorted(pred_df["date"].unique())
    date_labels = [pd.to_datetime(d).strftime("%m/%d") for d in dates_list]

    z = np.zeros((len(team_order), len(dates_list)))
    for i, team in enumerate(team_order):
        for j, date in enumerate(dates_list):
            val = pred_df[(pred_df["team"] == team) & (pred_df["date"] == date)]["prob_norm"]
            z[i, j] = val.values[0] if len(val) else np.nan

    y_labels = [f"{'★ ' if t in top5_teams else '   '}{t}" for t in team_order]
    text_z = [[f"{v:.0%}" for v in row] for row in z]

    fig = go.Figure(go.Heatmap(
        z=z,
        x=date_labels,
        y=y_labels,
        text=text_z,
        texttemplate="%{text}",
        textfont=dict(size=9),
        colorscale="Blues",
        zmin=0, zmax=1,
        hovertemplate="<b>%{y}</b><br>날짜: %{x}<br>확률: %{z:.1%}<extra></extra>",
        colorbar=dict(title="진출 확률", tickformat=".0%", len=0.8),
    ))

    # 컷라인 (4.5번째 y축)
    fig.add_hline(y=4.5, line_dash="dash", line_color=CUTLINE_COLOR,
                  line_width=1.5, opacity=0.8)

    tick_step = max(1, len(dates_list) // 8)
    fig.update_xaxes(
        tickvals=date_labels[::tick_step],
        ticktext=date_labels[::tick_step],
        tickangle=0,
    )
    fig.update_layout(
        plot_bgcolor=BG, paper_bgcolor=BG,
        margin=dict(l=10, r=10, t=20, b=40),
        height=380,
    )
    return fig
