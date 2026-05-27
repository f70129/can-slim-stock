# 📈 CAN SLIM 智能選股看板

William O'Neil CAN SLIM 法則 | 台股上市・上櫃 + 美股 | Streamlit Cloud

## 快速部署到 Streamlit Cloud

1. 將此 Repo fork 到自己帳號
2. 前往 share.streamlit.io → New app
3. 選擇 repo，Main file: `app.py`
4. 點 Deploy!

## 功能一覽

| 功能 | 說明 |
| 🔍 自動掃描 | 台股上市25支 / 上櫃20支 / 美股30支，CAN SLIM 評分排名 |
| 📊 前9名總覽表 | C/A/N/S/L/I/M 各項進度條，一目了然 |
| 📈 K線圖 | EMA21🟡(金粗) + MA50🔵 + MA200🩷 + 布林通道 + MACD + 成交量 |
| 🕸️ 雷達圖 | CAN SLIM 七維圖形化分析 |
| ⚡ 21日均線警示 | 收盤價在 EMA21 ±1.5% 內自動顯示橘/紫警示框 |
| 🔄 自動更新 | 每15分鐘重新抓取資料（需 streamlit-autorefresh） |
