# nasdaq_morning_app

NASDAQ-100 长线 + 短线综合分析工具（本地生成 HTML 报告）。

## 功能概览
- **短线候选评分**：多因子打分（动量、趋势、量能、资金流、风险等）
- **长线观点**：结合趋势、波动/回撤、新闻与未来事件线索
- **滚动胜率统计**：
  - 总胜率：前5个推荐日去重后统计（同一股票仅保留一次，取最早推荐价对比当前价）
  - 按日明细：使用该推荐日的推荐价对比当前价
- 输出 `nasdaq_morning_report.html` 并自动打开

## 模块结构（已拆分）
- `nasdaq_morning_app.py`：主入口（仅编排流程）
- `nasdaq_data_layer.py`：数据抓取与清洗（行情、新闻、事件）
- `nasdaq_strategy_layer.py`：策略评分、长短线计算、胜率统计
- `nasdaq_report_renderer.py`：HTML 页面渲染与图表输出
- `install_nasdaq_morning.sh`：本地安装辅助脚本

## 运行方式
```bash
python3 nasdaq_morning_app.py
```

若使用虚拟环境（推荐）：
```bash
./.venv-nasdaq/bin/python nasdaq_morning_app.py
```

## 输出文件
- `nasdaq_morning_report.html`：最新分析报告
- `nasdaq_reco_history.json`：推荐历史（用于滚动胜率统计）

## 注意事项
- 未来活动信息基于动态抓取（新闻/官网线索/数据源），不使用代码内写死事件。
- 结果仅供学习与研究，不构成任何投资建议。
