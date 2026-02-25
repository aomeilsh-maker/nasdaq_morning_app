from __future__ import annotations

import datetime as dt
import html
import json
import webbrowser
from pathlib import Path
from typing import Any, Dict, List


def render_report(picks: List[Any], regime: dict, winrate: Any, long_views: List[Any], x_status: Dict[str, Any]) -> None:
    generated_at = dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    risk_on = regime.get("risk_on", True)
    regime_text = (
        f"QQQ: {regime.get('price',0):.2f} | MA50: {regime.get('ma50',0):.2f} | MA200: {regime.get('ma200',0):.2f} | "
        f"市场状态: {'Risk-ON' if risk_on else 'Risk-OFF(降权)'}"
    )

    if winrate.total > 0:
        summary_text = f"前5个推荐日胜率：{winrate.wins}/{winrate.total} = {winrate.rate:.1f}%"
    else:
        summary_text = "前5个推荐日胜率：暂无可统计数据（需要先累积历史推荐）"

    if winrate.high_est_total > 0:
        high_est_text = (
            f"预估胜率≥{winrate.high_est_threshold:.0f}% 子集胜率："
            f"{winrate.high_est_wins}/{winrate.high_est_total} = {winrate.high_est_rate:.1f}%"
        )
    else:
        high_est_text = f"预估胜率≥{winrate.high_est_threshold:.0f}% 子集胜率：暂无可统计数据"

    winrate_items = "".join(
        f"<li>{html.escape(str(d['date']))}: {int(d['wins'])}/{int(d['total'])} = {float(d['rate']):.1f}%"
        + f"<br><span class='muted'>✅ 胜出: {html.escape(', '.join(d.get('win_symbols', [])) or '无')}</span>"
        + f"<br><span class='muted'>❌ 失利: {html.escape(', '.join(d.get('loss_symbols', [])) or '无')}</span></li>"
        for d in winrate.details
    ) or "<li>暂无明细</li>"

    labels = [p.symbol for p in picks]
    score_data = [round(float(p.score), 2) for p in picks]
    conf_data = [int(p.confidence) for p in picks]
    ret20_data = [round(float(p.ret_20d), 2) for p in picks]
    vol_data = [round(float(p.vol20_annual), 2) for p in picks]

    lt_labels = [v.symbol for v in long_views]
    lt_scores = [int(v.score) for v in long_views]
    long_cards = []
    for v in long_views:
        event_html = "".join(
            f"<li><b>{html.escape(e.get('type','事件'))}</b>："
            + (f"<a href='{html.escape(e.get('url',''), quote=True)}' target='_blank'>{html.escape(e.get('title','(no title)'))}</a>" if e.get('url') else html.escape(e.get('title','(no title)')))
            + f" <span class='muted'>[{html.escape(e.get('source',''))}]</span></li>"
            for e in v.events
        ) or "<li>暂无重大事件抓取</li>"

        long_cards.append(f"""
        <div class='card stock-card'>
          <div class='stock-head'>
            <h3>{html.escape(v.symbol)} <span class='muted'>· {html.escape(v.name)}</span></h3>
            <span class='badge {'grade-a' if v.score >= 72 else ('grade-b' if v.score >= 58 else 'grade-c')}>趋势分 {v.score}</span>
          </div>
          <div class='kv'>
            <span>现价 <b>{v.price:.2f}</b></span>
            <span>1年 <b>{v.ret_1y:.2f}%</b></span>
            <span>3年 <b>{v.ret_3y:.2f}%</b></span>
            <span>MA50/MA200 <b>{v.ma50:.1f}/{v.ma200:.1f}</b></span>
            <span>RSI14 <b>{v.rsi14:.1f}</b></span>
          </div>
          <div class='kv'>
            <span>年化波动 <b>{v.vol_1y:.1f}%</b></span>
            <span>1y最大回撤 <b>{v.max_dd_1y:.1f}%</b></span>
            <span>趋势判断 <b>{html.escape(v.trend_label)}</b></span>
          </div>
          <p class='muted'>本股消息池：主流 {v.news_pool_main} 条｜X {v.news_pool_x} 条｜最终采用 {v.news_used} 条</p>
          <p class='reason'><b>{html.escape(v.analysis)}</b></p>
          <ul class='analysis-list'>{''.join(f'<li>{html.escape(pt)}</li>' for pt in v.analysis_points)}</ul>
          <details><summary>消息面明细（逐条新闻与来源）</summary><ul>{''.join(f"<li>[{html.escape(str(n.get('source','')))}] <a href='{html.escape(str(n.get('url','')), quote=True)}' target='_blank'>{html.escape(str(n.get('title','(no title)')))}</a></li>" for n in v.long_news) or '<li>暂无新闻样本</li>'}</ul></details>
          <details><summary>未来3个月重大活动（未发生）</summary><ul>{event_html}</ul></details>
        </div>
        """)

    cards = []
    for i, p in enumerate(picks, start=1):
        if p.news:
            news_html = "".join(
                f"<li>[{html.escape(n.get('source','News'))}] <a href='{html.escape(n.get('url',''), quote=True)}' target='_blank'>{html.escape(n.get('title','(no title)'))}</a></li>"
                for n in p.news
            )
        else:
            news_html = "<li>暂无抓取到可用新闻</li>"

        cards.append(f"""
        <div class='card stock-card'>
          <div class='stock-head'>
            <h3>#{i} {html.escape(p.symbol)} <span class='muted'>· {html.escape(p.name)}</span></h3>
            <span class='badge grade-{p.grade.lower()}'>{p.grade} / {p.confidence}</span>
          </div>
          <div class='kv'>
            <span>现价 <b>{p.price:.2f}</b></span>
            <span>日涨跌 <b>{p.day_change:.2f}%</b></span>
            <span>5日 <b>{p.ret_5d:.2f}%</b></span>
            <span>20日 <b>{p.ret_20d:.2f}%</b></span>
            <span>60日 <b>{p.ret_60d:.2f}%</b></span>
          </div>
          <div class='kv'>
            <span>RSI14 <b>{p.rsi14:.1f}</b></span>
            <span>量比 <b>{p.volume_ratio:.2f}</b></span>
            <span>20D波动 <b>{p.vol20_annual:.1f}%</b></span>
            <span>60D回撤 <b>{p.max_dd_60d:.1f}%</b></span>
          </div>
          <p><b>资金流</b>：CMF20 {p.cmf20:.2f}｜OBV20 {p.obv_trend_20:.1f}%｜<b>{html.escape(p.accumulation_tag)}</b></p>
          <p><b>同信号5日胜率</b>：{p.signal_winrate_5d:.1f}%｜<b>情绪</b>：{html.escape(p.sentiment)}</p>
          <p><b>交易计划</b>：入场 {html.escape(p.entry_hint)}；止损 {html.escape(p.stop_hint)}；目标 {html.escape(p.target_hint)}</p>
          <p class='reason'>{html.escape(p.reason)}</p>
          <details><summary>相关新闻（含X）</summary><ul>{news_html}</ul></details>
        </div>
        """)

    if not cards:
        cards.append("<div class='card'><h3>无可用数据</h3><p>本次未抓到可用行情。</p></div>")

    page = f"""
    <!doctype html>
    <html lang='zh-CN'>
    <head>
      <meta charset='utf-8'/>
      <meta name='viewport' content='width=device-width, initial-scale=1'/>
      <title>NASDAQ-100 短线+长线综合分析 v4</title>
      <script src='https://cdn.jsdelivr.net/npm/chart.js'></script>
      <style>
        :root {{
          --bg:#0b1020; --panel:#121a2f; --panel2:#182441; --border:#2b3c67; --text:#e8ecf3; --muted:#9fb0d6;
          --a:#7fb3ff; --good:#2fd27a; --warn:#f8bf4a; --bad:#ff7070;
        }}
        * {{ box-sizing:border-box; }}
        body {{ margin:0; font-family:Inter,-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,sans-serif; color:var(--text);
               background: radial-gradient(1200px 700px at 20% -10%, #1a2950 0%, var(--bg) 45%); }}
        .wrap {{ max-width:1280px; margin:0 auto; padding:24px; }}
        .hero {{ background:linear-gradient(135deg,#1a2750,#0f1730); border:1px solid var(--border); border-radius:16px; padding:18px 20px; margin-bottom:14px; }}
        .hero h1 {{ margin:0 0 8px; font-size:28px; }}
        .meta {{ color:var(--muted); margin-bottom:8px; }}
        .pill {{ display:inline-block; padding:6px 12px; border-radius:999px; background:#1a2440; border:1px solid #37528f; }}
        .top-grid {{ display:grid; grid-template-columns: 1fr 1fr; gap:14px; margin-bottom:14px; }}
        .card {{ background:var(--panel); border:1px solid var(--border); border-radius:14px; padding:14px; box-shadow:0 8px 24px rgba(0,0,0,.18); }}
        .chart-box {{ height:280px; }}
        .stocks {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(390px,1fr)); gap:14px; }}
        .stock-head {{ display:flex; align-items:center; justify-content:space-between; gap:10px; }}
        .stock-head h3 {{ margin:0; font-size:18px; }}
        .muted {{ color:var(--muted); font-weight:500; font-size:13px; }}
        .badge {{ border-radius:10px; padding:4px 10px; font-weight:700; font-size:12px; border:1px solid transparent; }}
        .grade-a {{ background:rgba(47,210,122,.15); color:#74efad; border-color:rgba(47,210,122,.35); }}
        .grade-b {{ background:rgba(248,191,74,.15); color:#ffd88a; border-color:rgba(248,191,74,.35); }}
        .grade-c {{ background:rgba(255,112,112,.15); color:#ffb3b3; border-color:rgba(255,112,112,.35); }}
        .kv {{ display:grid; grid-template-columns:repeat(5,minmax(0,1fr)); gap:8px; margin:10px 0; color:var(--muted); font-size:13px; }}
        .kv span {{ display:flex; flex-direction:column; align-items:flex-start; gap:2px; }}
        .kv b {{ color:var(--text); }}
        .reason {{ color:#c8d6f5; margin-bottom:6px; }}
        .analysis-list {{ margin:6px 0 10px 18px; color:#d7e3ff; line-height:1.5; }}
        .analysis-list li {{ margin:4px 0; }}
        a {{ color:var(--a); }} summary {{ cursor:pointer; color:#c6d7ff; }}
        @media (max-width:980px) {{ .top-grid {{ grid-template-columns:1fr; }} .kv {{ grid-template-columns:repeat(2,minmax(0,1fr)); }} }}
      </style>
    </head>
    <body>
      <div class='wrap'>
        <section class='hero'>
          <h1>NASDAQ-100 短线 + 长线综合分析（v4）</h1>
          <div class='meta'>生成时间：{generated_at}｜仅供学习参考，不构成投资建议</div>
          <div class='pill'>{html.escape(regime_text)}</div>
        </section>

        <section class='card' style='margin:20px 0 14px 0; border:2px solid #f8bf4a;'>
          <h2 style='margin-top:0;'>📈 长线分析专区（INTC / NVDA / AMD / MSFT / AMZN / META / GOOGL）</h2>
          <p class='muted'>结合中长期趋势、波动/回撤与近期重大事件（财报、AI/产品、监管等）做定性分析。</p>
          <p class='muted'>X抓取状态：网页抓取可用 {x_status['ok']}/{x_status['total']}，X样本 {x_status['sample_count']}（仅在有主流媒体样本时作为辅助引用），模式 {x_status['mode']}。</p>
          <div class='chart-box'><canvas id='longTrendChart'></canvas></div>
        </section>

        <section class='stocks' style='margin-bottom:20px;'>{''.join(long_cards)}</section>

        <section class='card' style='margin:8px 0 14px 0; border:2px solid #7fb3ff;'>
          <h2 style='margin:0;'>⚡ 短线分析专区（今日候选）</h2>
          <p class='muted'>以下是日内/短周期交易信号排序结果。</p>
        </section>

        <section class='top-grid'>
          <div class='card'>
            <h3>前五日推荐股票胜率（滚动）</h3>
            <p><b>{html.escape(summary_text)}</b></p>
            <p><b>{html.escape(high_est_text)}</b></p>
            <details><summary>查看按推荐日明细</summary><ul>{winrate_items}</ul></details>
          </div>
          <div class='card'>
            <h3>当日Top候选评分与置信度</h3>
            <div class='chart-box'><canvas id='scoreChart'></canvas></div>
          </div>
        </section>

        <section class='card' style='margin-bottom:14px;'>
          <h3>20日收益 vs 年化波动（风险收益散点）</h3>
          <div class='chart-box'><canvas id='rvChart'></canvas></div>
        </section>

        <section class='stocks'>{''.join(cards)}</section>
      </div>

      <script>
        const labels = {json.dumps(labels, ensure_ascii=False)};
        const scoreData = {json.dumps(score_data, ensure_ascii=False)};
        const confData = {json.dumps(conf_data, ensure_ascii=False)};
        const ret20Data = {json.dumps(ret20_data, ensure_ascii=False)};
        const volData = {json.dumps(vol_data, ensure_ascii=False)};
        const ltLabels = {json.dumps(lt_labels, ensure_ascii=False)};
        const ltScores = {json.dumps(lt_scores, ensure_ascii=False)};

        new Chart(document.getElementById('longTrendChart'), {{
          type: 'bar',
          data: {{
            labels: ltLabels,
            datasets: [{{ label: '长线趋势分', data: ltScores, backgroundColor: 'rgba(248,191,74,.6)', borderColor:'#f8bf4a', borderWidth:1 }}]
          }},
          options: {{ responsive:true, maintainAspectRatio:false,
            plugins: {{ legend: {{ labels: {{ color:'#dbe6ff' }} }} }},
            scales: {{
              x: {{ ticks: {{ color:'#c5d4f8' }}, grid: {{ color:'rgba(140,160,210,.15)' }} }},
              y: {{ min:0, max:100, ticks: {{ color:'#c5d4f8' }}, grid: {{ color:'rgba(140,160,210,.15)' }} }}
            }}
          }}
        }});

        new Chart(document.getElementById('scoreChart'), {{
          type: 'bar',
          data: {{
            labels,
            datasets: [
              {{ label: '综合评分', data: scoreData, backgroundColor: 'rgba(127,179,255,.65)', borderColor:'#7fb3ff', borderWidth:1 }},
              {{ label: '置信度', data: confData, type:'line', yAxisID:'y1', borderColor:'#2fd27a', backgroundColor:'rgba(47,210,122,.2)', tension:.25 }}
            ]
          }},
          options: {{ responsive:true, maintainAspectRatio:false,
            plugins: {{ legend: {{ labels: {{ color:'#dbe6ff' }} }} }},
            scales: {{
              x: {{ ticks: {{ color:'#c5d4f8' }}, grid: {{ color:'rgba(140,160,210,.15)' }} }},
              y: {{ ticks: {{ color:'#c5d4f8' }}, grid: {{ color:'rgba(140,160,210,.15)' }} }},
              y1: {{ position:'right', min:50, max:100, ticks: {{ color:'#9ce4bb' }}, grid: {{ drawOnChartArea:false }} }}
            }}
          }}
        }});

        new Chart(document.getElementById('rvChart'), {{
          type:'scatter',
          data: {{ datasets: labels.map((s,i)=>({{ label:s, data:[{{x:volData[i], y:ret20Data[i]}}], pointRadius:6 }})) }},
          options: {{ responsive:true, maintainAspectRatio:false,
            plugins: {{ legend: {{ labels: {{ color:'#dbe6ff' }} }} }},
            scales: {{
              x: {{ title: {{ display:true, text:'年化波动(20D)%', color:'#dbe6ff' }}, ticks: {{ color:'#c5d4f8' }}, grid: {{ color:'rgba(140,160,210,.15)' }} }},
              y: {{ title: {{ display:true, text:'20日收益%', color:'#dbe6ff' }}, ticks: {{ color:'#c5d4f8' }}, grid: {{ color:'rgba(140,160,210,.15)' }} }}
            }}
          }}
        }});
      </script>
    </body>
    </html>
    """

    out = Path(__file__).with_name("nasdaq_morning_report.html")
    out.write_text(page, encoding="utf-8")
    webbrowser.open(out.resolve().as_uri())

