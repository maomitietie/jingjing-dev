import streamlit as st
import requests
import json
import sqlite3
import pandas as pd
from datetime import datetime, date
import matplotlib.pyplot as plt
import matplotlib
import io
import csv
import re

# ============================================================
# 数据库操作（增加运动、心情字段）
# ============================================================

def init_db():
    """初始化数据库，创建 health_records 表（包含运动、心情）"""
    conn = sqlite3.connect("health_assassin.db")
    c = conn.cursor()
    # 尝试创建新表，如果已存在则后续通过 ALTER 添加列（兼容旧库）
    c.execute("""
        CREATE TABLE IF NOT EXISTS health_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            date TEXT NOT NULL,
            sit TEXT,
            head TEXT,
            sleep REAL,
            water TEXT,
            stress INTEGER,
            weather TEXT,
            temperature TEXT,
            ai_response TEXT,
            comprehensive_score INTEGER,
            exercise TEXT,
            mood TEXT,
            UNIQUE(username, date)
        )
    """)
    # 检查并添加缺失的列（兼容旧数据库）
    c.execute("PRAGMA table_info(health_records)")
    columns = [col[1] for col in c.fetchall()]
    if "exercise" not in columns:
        c.execute("ALTER TABLE health_records ADD COLUMN exercise TEXT")
    if "mood" not in columns:
        c.execute("ALTER TABLE health_records ADD COLUMN mood TEXT")
    conn.commit()
    conn.close()


def save_record(username, data):
    """保存或更新（覆盖）用户的当天健康记录（增加运动、心情）"""
    conn = sqlite3.connect("health_assassin.db")
    c = conn.cursor()
    today = date.today().isoformat()
    c.execute("""
        INSERT INTO health_records (username, date, sit, head, sleep, water, stress, weather, temperature, ai_response, comprehensive_score, exercise, mood)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(username, date) DO UPDATE SET
            sit = excluded.sit,
            head = excluded.head,
            sleep = excluded.sleep,
            water = excluded.water,
            stress = excluded.stress,
            weather = excluded.weather,
            temperature = excluded.temperature,
            ai_response = excluded.ai_response,
            comprehensive_score = excluded.comprehensive_score,
            exercise = excluded.exercise,
            mood = excluded.mood
    """, (
        username, today,
        data.get("sit"), data.get("head"), data.get("sleep"),
        data.get("water"), data.get("stress"),
        data.get("weather"), data.get("temperature"),
        data.get("ai_response"), data.get("comprehensive_score"),
        data.get("exercise"), data.get("mood")
    ))
    conn.commit()
    conn.close()


def get_user_records(username, limit=5):
    """获取指定用户的历史记录，按日期降序（包含运动、心情）"""
    conn = sqlite3.connect("health_assassin.db")
    c = conn.cursor()
    c.execute("""
        SELECT date, sit, head, sleep, water, stress, weather, temperature, ai_response, comprehensive_score, exercise, mood
        FROM health_records
        WHERE username = ?
        ORDER BY date DESC
        LIMIT ?
    """, (username, limit))
    rows = c.fetchall()
    conn.close()
    return rows


def get_user_records_all(username):
    """获取指定用户的所有历史记录（用于导出和绘图）"""
    conn = sqlite3.connect("health_assassin.db")
    c = conn.cursor()
    c.execute("""
        SELECT date, sit, head, sleep, water, stress, weather, temperature, ai_response, comprehensive_score, exercise, mood
        FROM health_records
        WHERE username = ?
        ORDER BY date DESC
    """, (username,))
    rows = c.fetchall()
    conn.close()
    return rows


def get_recent_scores(username, days=7):
    """获取最近 N 天的综合健康指数（用于趋势图），按日期升序"""
    conn = sqlite3.connect("health_assassin.db")
    c = conn.cursor()
    c.execute("""
        SELECT date, comprehensive_score
        FROM health_records
        WHERE username = ?
        ORDER BY date DESC
        LIMIT ?
    """, (username, days))
    rows = c.fetchall()
    conn.close()
    rows.reverse()  # 按日期升序
    return rows


def delete_record(username, record_date):
    """删除指定用户的某条记录"""
    conn = sqlite3.connect("health_assassin.db")
    c = conn.cursor()
    c.execute("DELETE FROM health_records WHERE username = ? AND date = ?", (username, record_date))
    conn.commit()
    deleted = c.rowcount > 0
    conn.close()
    return deleted


def get_average_score(username, days=7):
    """获取最近 N 天的平均综合健康指数"""
    scores = get_recent_scores(username, days)
    if not scores:
        return None
    valid_scores = [s[1] for s in scores if s[1] is not None]
    if not valid_scores:
        return None
    avg = sum(valid_scores) / len(valid_scores)
    return round(avg, 1)


def get_all_usernames():
    """获取数据库中所有不同的用户名"""
    conn = sqlite3.connect("health_assassin.db")
    c = conn.cursor()
    c.execute("SELECT DISTINCT username FROM health_records")
    rows = c.fetchall()
    conn.close()
    return [row[0] for row in rows]


def get_latest_records_sidebar(username, limit=3):
    """获取最近几条记录，用于侧边栏显示"""
    rows = get_user_records(username, limit)
    return rows


# ============================================================
# 天气获取（不变）
# ============================================================

def get_weather(city):
    """调用 wttr.in 获取城市天气，返回 (状况, 气温, 湿度)"""
    try:
        url = f"https://wttr.in/{city}?format=%C|%t|%h"
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            parts = resp.text.strip().split("|")
            condition = parts[0] if len(parts) > 0 else "未知"
            temp = parts[1] if len(parts) > 1 else "未知"
            humidity = parts[2] if len(parts) > 2 else "未知"
            return condition, temp, humidity
        else:
            return "未知", "未知", "未知"
    except Exception:
        return "未知", "未知", "未知"


# ============================================================
# DeepSeek API 调用（支持对话历史）
# ============================================================

def call_deepseek(api_key, messages):
    """调用 DeepSeek API，messages 为对话列表"""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "deepseek-chat",
        "messages": messages,
        "max_tokens": 1024
    }
    try:
        r = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=10
        )
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"]
        else:
            return f"API 请求失败（{r.status_code}）：{r.text}"
    except requests.exceptions.Timeout:
        return "请求超时，请检查网络连接后重试。"
    except Exception as e:
        return f"请求出错：{str(e)}"


def build_prompt(sit, head, sleep, water, stress, exercise, mood, weather_info):
    """根据用户输入和天气构建 prompt（增加运动和心情）"""
    weather_str = f"当地天气：{weather_info['condition']}，气温{weather_info['temp']}，湿度{weather_info['humidity']}" if weather_info else "天气信息未知"
    return f"""
我是一名关注健康的大学生，以下是我今天的生活数据：

- 久坐最长时段：{sit}
- 低头累计时长：{head}
- 睡眠时长：{sleep}小时
- 今日饮水量：{water}
- 今日压力自评：{stress}/10分
- 今日运动时长：{exercise}
- 今日心情：{mood}
- {weather_str}

请严格按照以下四个部分回答，每个部分用分隔线 "---" 隔开：

第一部分：反击动作
给出一个 2-5 分钟可完成的动作，必须结合上述天气和生活数据（特别是运动和心情），动作要具体到步骤。

第二部分：生活健康评分表
久坐健康分：xx/100（说明）
颈椎健康分：xx/100（说明）
睡眠质量分：xx/100（说明）
水分充足分：xx/100（说明）
压力管理分：xx/100（说明）
运动活跃分：xx/100（说明）
综合健康指数：xx/100

第三部分：刺客有话说
一段像朋友聊天一样的自然语言评语（2-3句话），风格温暖+轻微幽默，结合天气、睡眠、压力、心情等指标。

第四部分：明日小贴士
针对今天最薄弱的一项指标，给出一个明天可以做的具体改进建议。
"""


def parse_score(text):
    """从 AI 回复中解析综合健康指数"""
    try:
        for line in text.split("\n"):
            if "综合健康指数" in line:
                nums = re.findall(r'\d+', line)
                if nums:
                    score = int(nums[0])
                    return min(max(score, 0), 100)
    except Exception:
        pass
    return None


# ============================================================
# 随机健康小贴士（用于加载时展示）
# ============================================================

import random

HEALTH_TIPS = [
    "🌱 每坐45分钟站起来走动2分钟，能有效降低腰椎压力！",
    "💡 每天喝够1.5L水，大脑反应速度提升约14%！",
    "🌟 午间小憩20分钟，比喝咖啡更能提升下午工作效率！",
    "🌈 深呼吸10次（每次5秒），焦虑感立即降低30%！",
    "🔥 每天晒15分钟太阳，维生素D合成效率提升90%！",
    "💪 靠墙站立5分钟，矫正体态的同时消耗约20大卡！",
    "🧠 每周3次有氧运动，记忆力提升约20%！",
    "🌿 保持7-8小时睡眠，免疫力提升3倍！",
    "⭐ 每小时转动脖颈30秒，颈椎病风险降低50%！",
    "🍃 看绿植1分钟，眼部疲劳缓解效果等同远眺10分钟！",
    "✨ 饭后散步10分钟，血糖波动减少约30%！",
    "🎯 每天大笑10次，相当于中等强度有氧运动5分钟！",
    "🌸 工作间隙拉伸肩胛骨，圆肩驼背改善率提升60%！",
    "💧 温水+柠檬早起喝，新陈代谢提速约10%！",
    "🌻 保持正确坐姿：腰背贴椅背，视线平视屏幕上方！",
]


def get_random_tip():
    """获取一条随机健康小贴士"""
    return random.choice(HEALTH_TIPS)


def get_loading_phrases():
    """返回加载阶段的趣味短语列表"""
    return [
        ("🔍 正在扫描你的健康数据...", "收集久坐、低头、睡眠等指标"),
        ("📊 正在分析生活习惯...", "结合天气和运动数据进行评估"),
        ("🧠 AI正在生成个性化方案...", "量身定制你的反击动作"),
        ("⚡ 正在计算综合健康指数...", "各项指标综合评分中"),
        ("🎯 正在生成健康报告...", "马上就能看到结果啦！"),
    ]


# ============================================================
# 图表绘制（不变）
# ============================================================

def plot_trend_chart(scores):
    """绘制综合健康指数趋势折线图"""
    matplotlib.use("Agg")
    fig, ax = plt.subplots(figsize=(8, 3))

    if not scores:
        ax.text(0.5, 0.5, "暂无数据", ha="center", va="center", fontsize=14)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        return fig

    dates = [s[0] for s in scores]
    score_values = [s[1] if s[1] is not None else 0 for s in scores]

    ax.plot(dates, score_values, marker="o", color="#FF6B6B", linewidth=2, markersize=6)
    ax.fill_between(dates, score_values, alpha=0.2, color="#FF6B6B")

    ax.set_ylim(0, 100)
    ax.set_ylabel("综合健康指数")
    ax.set_title("最近 7 天健康趋势", fontsize=14, fontweight="bold")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


# ============================================================
# CSV 导出（增加运动、心情）
# ============================================================

def export_to_csv(username):
    """导出当前用户的所有记录为 CSV 字符串"""
    records = get_user_records_all(username)
    if not records:
        return None

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["日期", "久坐", "低头", "睡眠(小时)", "饮水量", "压力评分",
                     "运动时长", "心情", "天气", "气温", "综合健康指数", "AI回复"])
    for r in records:
        writer.writerow([r[0], r[1], r[2], r[3], r[4], r[5],
                         r[10], r[11], r[6], r[7], r[9], r[8]])
    return output.getvalue()


# ============================================================
# Streamlit 主页面
# ============================================================

def main():
    st.set_page_config(page_title="五分钟健康刺客", layout="wide")

    # ========== 自定义CSS动画样式 ==========
    st.markdown("""
    <style>
        /* 全局动画 */
        @keyframes fadeInUp {
            from { opacity: 0; transform: translateY(30px); }
            to { opacity: 1; transform: translateY(0); }
        }
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.05); }
            100% { transform: scale(1); }
        }
        @keyframes float {
            0% { transform: translateY(0px); }
            50% { transform: translateY(-10px); }
            100% { transform: translateY(0px); }
        }
        @keyframes glow {
            0% { box-shadow: 0 0 5px rgba(255,107,107,0.3); }
            50% { box-shadow: 0 0 20px rgba(255,107,107,0.6); }
            100% { box-shadow: 0 0 5px rgba(255,107,107,0.3); }
        }
        @keyframes shimmer {
            0% { background-position: -200% center; }
            100% { background-position: 200% center; }
        }
        @keyframes spin-slow {
            from { transform: rotate(0deg); }
            to { transform: rotate(360deg); }
        }
        @keyframes bounce-in {
            0% { opacity: 0; transform: scale(0.3); }
            50% { transform: scale(1.1); }
            70% { transform: scale(0.9); }
            100% { opacity: 1; transform: scale(1); }
        }
        @keyframes gradient-shift {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        /* 标题动画 */
        .main-title {
            font-size: 2.8rem;
            font-weight: 800;
            background: linear-gradient(135deg, #FF6B6B, #FF8E53, #FF6B6B, #ee5a6f);
            background-size: 300% 300%;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            animation: gradient-shift 4s ease infinite, fadeInUp 0.8s ease-out;
            text-align: center;
            padding: 20px 0 10px 0;
        }
        .main-title-sub {
            text-align: center;
            color: #888;
            font-size: 1rem;
            animation: fadeInUp 1s ease-out;
            margin-bottom: 20px;
        }

        /* 卡片容器 */
        .card-container {
            background: linear-gradient(135deg, #ffffff, #f8f9fa);
            border-radius: 16px;
            padding: 20px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.08);
            border: 1px solid rgba(255,107,107,0.1);
            animation: fadeInUp 0.6s ease-out;
            transition: all 0.3s ease;
            margin-bottom: 16px;
        }
        .card-container:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 25px rgba(255,107,107,0.15);
            border-color: rgba(255,107,107,0.3);
        }

        /* 浮动装饰元素 */
        .float-emoji {
            display: inline-block;
            animation: float 3s ease-in-out infinite;
            font-size: 1.5rem;
        }
        .pulse-emoji {
            display: inline-block;
            animation: pulse 2s ease-in-out infinite;
        }

        /* 健康指数徽章 */
        .score-badge {
            display: inline-block;
            padding: 8px 20px;
            border-radius: 50px;
            font-size: 1.3rem;
            font-weight: 700;
            animation: bounce-in 0.8s ease-out;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        .score-badge.excellent { background: linear-gradient(135deg, #56CCF2, #2F80ED); color: white; }
        .score-badge.good { background: linear-gradient(135deg, #A8E6CF, #7EC8A8); color: white; }
        .score-badge.fair { background: linear-gradient(135deg, #FFEAA7, #FDCB6E); color: #333; }
        .score-badge.poor { background: linear-gradient(135deg, #FFB3B3, #FF6B6B); color: white; }

        /* 加载动画容器 */
        .loading-container {
            background: linear-gradient(135deg, #667eea22, #764ba222);
            border-radius: 16px;
            padding: 24px;
            border: 2px dashed #667eea44;
            animation: glow 2s ease-in-out infinite;
        }
        .loading-tip {
            font-size: 1.1rem;
            color: #555;
            padding: 10px;
            border-left: 4px solid #FF6B6B;
            background: #FFF5F5;
            border-radius: 0 8px 8px 0;
            margin-top: 12px;
        }

        /* 侧边栏美化 */
        .sidebar-user-card {
            background: linear-gradient(135deg, #667eea22, #764ba222);
            border-radius: 12px;
            padding: 12px;
            text-align: center;
            border: 1px solid #667eea33;
            margin: 8px 0;
        }

        /* 天气卡片 */
        .weather-card {
            background: linear-gradient(135deg, #74b9ff33, #a29bfe33);
            border-radius: 12px;
            padding: 12px 16px;
            text-align: center;
            animation: fadeInUp 0.8s ease-out;
        }
        .weather-card .temp {
            font-size: 2rem;
            font-weight: 700;
            color: #2d3436;
        }

        /* 按钮脉冲 */
        div.stButton > button {
            transition: all 0.3s ease !important;
        }
        div.stButton > button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 20px rgba(255,107,107,0.3) !important;
        }

        /* 进度条动画 */
        .stProgress > div > div > div > div {
            background: linear-gradient(90deg, #FF6B6B, #FFEAA7, #56CCF2, #FF6B6B) !important;
            background-size: 300% 100% !important;
            animation: shimmer 2s linear infinite !important;
        }

        /* 指标输入区域 */
        .indicator-group {
            animation: fadeInUp 0.6s ease-out;
        }

        /* 记录卡片 */
        .record-card {
            transition: all 0.3s ease;
        }
        .record-card:hover {
            transform: translateX(5px);
        }

        /* 聊天消息 */
        .chat-msg {
            animation: fadeInUp 0.4s ease-out;
            padding: 8px 12px;
            border-radius: 12px;
            margin: 4px 0;
        }
        .chat-msg.user {
            background: #667eea22;
            border-left: 3px solid #667eea;
        }
        .chat-msg.assistant {
            background: #FF6B6B11;
            border-left: 3px solid #FF6B6B;
        }

        /* 滚动条美化 */
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: #f1f1f1; border-radius: 10px; }
        ::-webkit-scrollbar-thumb { background: #FF6B6B88; border-radius: 10px; }
        ::-webkit-scrollbar-thumb:hover { background: #FF6B6B; }
    </style>
    """, unsafe_allow_html=True)

    # 显示带动画的标题
    st.markdown('<div class="main-title">⚔ 五分钟健康刺客</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-title-sub">💪 每天5分钟，给健康来一次精准反击！</div>', unsafe_allow_html=True)

    # 初始化数据库
    init_db()

    # 初始化 session_state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "last_username" not in st.session_state:
        st.session_state.last_username = ""
    if "last_ai_response" not in st.session_state:
        st.session_state.last_ai_response = None
    if "last_score" not in st.session_state:
        st.session_state.last_score = None

    # ========== 侧边栏 ==========
    with st.sidebar:
        st.header("⚙ 设置")

        api_key = st.text_input("DeepSeek API Key", type="password",
                                help="输入你的 DeepSeek API 密钥")

        st.subheader("🌍 常驻城市")
        preset_cities = {"湛江": "湛江", "佛山": "佛山", "自定义": ""}
        selected_city_preset = st.selectbox("选择常驻城市", list(preset_cities.keys()), index=0, help="快速选择湛江或佛山")

        if selected_city_preset == "自定义":
            city = st.text_input("输入城市", value="北京", help="自定义城市名")
        else:
            city = preset_cities[selected_city_preset]
            st.success(f"✅ 已选择常驻城市：{city}")

        enable_weather = st.checkbox("启用天气功能", value=True)

        st.markdown("---")
        st.subheader("👤 用户")

        all_users = get_all_usernames()
        selected_user = st.selectbox("快速选择已有用户", options=[""] + all_users, help="从下拉列表选择已有用户")
        if selected_user:
            username_input = selected_user
        else:
            username_input = st.text_input("或输入新用户名", value="默认用户")

        username = username_input.strip()
        if not username:
            username = "默认用户"

        # 用户卡片
        st.markdown(f"""
        <div class="sidebar-user-card">
            <div style="font-size:2rem;">👤</div>
            <div style="font-weight:700; font-size:1.1rem;">{username}</div>
            <div style="font-size:0.8rem; color:#666;">当前在线</div>
        </div>
        """, unsafe_allow_html=True)

        if st.session_state.last_username != username:
            st.session_state.chat_history = []
            st.session_state.last_ai_response = None
            st.session_state.last_score = None
            st.session_state.last_username = username
            st.info(f"已切换到用户：{username}")

        st.markdown("---")
        st.subheader("📋 近期记录")
        recent_records = get_latest_records_sidebar(username, 3)
        if recent_records:
            for rec in recent_records:
                rec_date = rec[0]
                rec_score = rec[9] if rec[9] is not None else "?"
                st.markdown(f"""
                <div style="background:#f8f9fa;border-radius:8px;padding:8px;margin:4px 0;font-size:0.9rem;">
                    📅 {rec_date} &nbsp; 健康指数：<strong>{rec_score}</strong>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.caption("暂无记录")

        st.markdown("---")
        st.subheader("📈 健康趋势")
        recent_scores = get_recent_scores(username, 7)
        if recent_scores:
            avg = get_average_score(username, 7)
            if avg is not None:
                st.info(f"📊 近7天平均综合健康指数：**{avg}**")
        fig = plot_trend_chart(recent_scores)
        st.pyplot(fig)

        # 趣味标语
        st.markdown("""
        <div style="text-align:center; opacity:0.6; font-size:0.8rem; margin-top:8px;">
            ⚔ 每天5分钟 · 健康不放松
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        if st.button("📥 导出历史记录 (CSV)"):
            csv_data = export_to_csv(username)
            if csv_data:
                st.download_button(
                    label="下载 CSV",
                    data=csv_data,
                    file_name=f"{username}_健康记录.csv",
                    mime="text/csv"
                )
            else:
                st.warning("暂无数据可导出")

    # ========== 主页面 ==========
    st.subheader(f"👋 {username}，今天感觉怎么样？")

    # 获取天气（增强显示）
    weather_info = None
    if enable_weather and city:
        condition, temp, humidity = get_weather(city)
        if condition != "未知":
            weather_info = {"condition": condition, "temp": temp, "humidity": humidity}
            # 美化天气卡片
            weather_icon_map = {
                "晴": "☀️", "晴间多云": "🌤", "多云": "⛅", "阴": "☁️",
                "雨": "🌧", "小雨": "🌦", "中雨": "🌧", "大雨": "🌧",
                "雪": "❄️", "雾": "🌫", "霾": "🌫", "风": "🌬"
            }
            icon = "🌡"
            for key, emoji in weather_icon_map.items():
                if key in condition:
                    icon = emoji
                    break
            st.markdown(f"""
            <div class="weather-card" style="margin-bottom:16px;">
                <div style="font-size:0.9rem; color:#636e72;">📍 {city} · 实时天气</div>
                <div style="display:flex; justify-content:center; align-items:center; gap:20px; padding:8px 0;">
                    <div style="font-size:3rem;">{icon}</div>
                    <div>
                        <div class="temp">{temp}</div>
                        <div style="color:#636e72;">{condition}</div>
                    </div>
                    <div style="color:#636e72; font-size:0.9rem;">
                        💧 湿度 {humidity}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("⚠ 天气获取失败，使用默认值继续")
            weather_info = {"condition": "未知", "temp": "未知", "humidity": "未知"}
    else:
        weather_info = {"condition": "未知", "temp": "未知", "humidity": "未知"}

    # 随机健康小贴士（带浮动动画）
    st.markdown(f"""
    <div class="card-container" style="padding:12px 20px; margin-bottom:16px;">
        <div style="display:flex; align-items:center; gap:12px;">
            <span class="float-emoji" style="font-size:2rem;">💡</span>
            <div>
                <div style="font-size:0.8rem; color:#888;">今日健康小贴士</div>
                <div style="font-weight:500;">{get_random_tip()}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 七项健康指标输入（原有5项 + 运动 + 心情）
    st.markdown('<div class="card-container"><div style="font-size:1.1rem; font-weight:600; margin-bottom:12px;">📊 今日健康指标</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        sit = st.radio("🪑 久坐最长时段",
                       ["<1小时", "1-3小时", "3-5小时", ">5小时"],
                       index=1, horizontal=True)
        head = st.radio("📱 低头累计时长",
                        ["<1小时", "1-3小时", "3-5小时", ">5小时"],
                        index=1, horizontal=True)
        water = st.radio("💧 今日饮水量",
                         ["<500ml", "500-1000ml", "1000-1500ml", ">1500ml"],
                         index=1, horizontal=True)
        # 新增：运动时长
        exercise = st.radio("🏃 今日运动时长",
                            ["<15分钟", "15-30分钟", "30-60分钟", ">60分钟"],
                            index=1, horizontal=True)

    with col2:
        sleep = st.slider("😴 睡眠时长（小时）",
                          min_value=3.0, max_value=10.0, value=7.0, step=0.5)
        stress = st.slider("😰 今日压力自评",
                           min_value=1, max_value=10, value=5, step=1,
                           help="1=无压力，10=极大压力")
        # 新增：心情
        mood_options = {"😊 开心": "开心", "😐 平静": "平静", "😞 低落": "低落", "😠 烦躁": "烦躁", "😴 疲惫": "疲惫"}
        mood_selected = st.selectbox("😊 今日心情", list(mood_options.keys()), index=1)
        mood = mood_options[mood_selected]

    st.markdown('</div>', unsafe_allow_html=True)

    # "召唤反击" 按钮
    if st.button("⚔ 召唤反击", type="primary", use_container_width=True):
        if not api_key:
            st.error("❌ 请先在侧边栏输入 DeepSeek API Key")
        else:
            # ===== 趣味多步骤加载动画 =====
            loading_phrases = get_loading_phrases()
            random_tip = get_random_tip()

            # 使用 st.status 创建步骤式加载
            status_placeholder = st.status("� 健康刺客正在集结力量...", expanded=True)

            # 显示随机健康小贴士
            status_placeholder.markdown(f"💡 **健康小贴士：** {random_tip}")

            # 步骤1
            status_placeholder.write(f"{loading_phrases[0][0]}  {loading_phrases[0][1]}")
            progress_bar = st.progress(0)
            import time
            time.sleep(0.5)
            progress_bar.progress(15)

            # 步骤2
            status_placeholder.write(f"{loading_phrases[1][0]}  {loading_phrases[1][1]}")
            progress_bar.progress(30)
            time.sleep(0.3)

            # 步骤3 - 真正的 API 调用
            status_placeholder.write(f"{loading_phrases[2][0]}  {loading_phrases[2][1]}")
            progress_bar.progress(50)

            prompt = build_prompt(sit, head, sleep, water, stress, exercise, mood, weather_info)
            messages = [{"role": "user", "content": prompt}]

            # 在调 API 前显示一个趣味等待
            status_placeholder.write("🤖 **AI正在思考...** 这需要几秒钟，先活动一下脖子吧！")

            ai_response = call_deepseek(api_key, messages)

            # 步骤4
            progress_bar.progress(75)
            status_placeholder.write(f"{loading_phrases[3][0]}  {loading_phrases[3][1]}")
            time.sleep(0.3)

            # 步骤5 - 完成
            progress_bar.progress(100)
            status_placeholder.write(f"{loading_phrases[4][0]}  ✅ {loading_phrases[4][1]}")
            status_placeholder.update(label="✅ 健康报告已生成！", state="complete", expanded=False)
            progress_bar.empty()

            score = parse_score(ai_response)
            if score is None:
                score = 75

            record_data = {
                "sit": sit,
                "head": head,
                "sleep": sleep,
                "water": water,
                "stress": stress,
                "exercise": exercise,
                "mood": mood,
                "weather": weather_info["condition"] if weather_info else "未知",
                "temperature": weather_info["temp"] if weather_info else "未知",
                "ai_response": ai_response,
                "comprehensive_score": score
            }
            try:
                save_record(username, record_data)
            except Exception as e:
                st.warning(f"⚠ 记录保存失败：{e}")

            st.session_state.last_ai_response = ai_response
            st.session_state.last_score = score
            st.rerun()

    # 显示上一次的AI结果
    if st.session_state.last_ai_response:
        st.markdown("---")
        st.markdown('<div class="card-container">', unsafe_allow_html=True)

        # 显示分数徽章
        if hasattr(st.session_state, 'last_score') and st.session_state.last_score:
            score = st.session_state.last_score
            if score >= 80:
                badge_class = "excellent"
                badge_icon = "🌟"
                badge_text = "优秀"
            elif score >= 60:
                badge_class = "good"
                badge_icon = "👍"
                badge_text = "良好"
            elif score >= 40:
                badge_class = "fair"
                badge_icon = "⚠️"
                badge_text = "一般"
            else:
                badge_class = "poor"
                badge_icon = "⚡"
                badge_text = "需关注"

            col_score, col_report = st.columns([1, 3])
            with col_score:
                st.markdown(f"""
                <div style="text-align:center; padding: 20px 10px;">
                    <div class="score-badge {badge_class}" style="font-size:2rem;">
                        {badge_icon} {score}
                    </div>
                    <div style="margin-top:8px; font-size:0.9rem; color:#666;">{badge_text}</div>
                </div>
                """, unsafe_allow_html=True)
            with col_report:
                st.markdown("### 💊 健康诊断报告")
        else:
            st.markdown("### 💊 健康诊断报告")

        st.markdown(st.session_state.last_ai_response)
        st.markdown('</div>', unsafe_allow_html=True)

    # ========== 健康 Chatbot ==========
    st.markdown("---")
    st.markdown('<div class="card-container">', unsafe_allow_html=True)
    st.markdown("""
    <div style="display:flex; align-items:center; gap:8px; margin-bottom:12px;">
        <span class="pulse-emoji" style="font-size:1.8rem;">💬</span>
        <span style="font-size:1.2rem; font-weight:600;">健康小助手</span>
        <span style="font-size:0.8rem; color:#888;">问我任何健康问题</span>
    </div>
    """, unsafe_allow_html=True)

    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f'<div class="chat-msg user"><strong>🧑 你：</strong>{msg["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-msg assistant"><strong>🏥 助手：</strong>{msg["content"]}</div>', unsafe_allow_html=True)

    user_question = st.text_input("💭 输入你的健康问题：", key="health_question", placeholder="例如：久坐后如何放松腰部？熬夜怎么补救？")
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 5])
    with col_btn1:
        if st.button("📩 发送", key="ask_btn"):
            if not api_key:
                st.error("❌ 请先在侧边栏输入 DeepSeek API Key")
            elif user_question.strip():
                st.session_state.chat_history.append({"role": "user", "content": user_question})
                context = st.session_state.chat_history[-10:]
                messages = [{"role": "system", "content": "你是一位温暖、专业的健康助手，专注于回答大学生常见的健康问题（久坐、熬夜、压力、饮食、运动等）。回答要简洁、亲切、有科学依据。"}]
                messages.extend(context)
                with st.spinner("🤔 思考中..."):
                    reply = call_deepseek(api_key, messages)
                st.session_state.chat_history.append({"role": "assistant", "content": reply})
                st.rerun()
            else:
                st.warning("请输入问题")

    with col_btn2:
        if st.button("🧹 清空"):
            st.session_state.chat_history = []
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

    # ========== 历史记录展示（主页面下方） ==========
    st.markdown("---")
    st.markdown(f'<div style="display:flex; align-items:center; gap:12px; margin-bottom:16px;"><span class="pulse-emoji" style="font-size:1.8rem;">📋</span><span style="font-size:1.3rem; font-weight:600;">{username} 的近 5 条记录</span></div>', unsafe_allow_html=True)
    records = get_user_records(username, 5)

    if not records:
        st.info("还没有健康记录，开始你的第一次「召唤反击」吧！")
    else:
        col_stat1, col_stat2, col_stat3 = st.columns(3)
        with col_stat1:
            avg_all = get_average_score(username, 7)
            if avg_all is not None:
                st.metric("📊 近7天平均健康指数", avg_all, delta=None)
        with col_stat2:
            if records and records[0][9] is not None:
                st.metric("🏆 最新健康指数", records[0][9])
        with col_stat3:
            total = len(get_user_records_all(username))
            st.metric("📚 记录总条数", total)

        for i, rec in enumerate(records):
            rec_date, rec_sit, rec_head, rec_sleep, rec_water, rec_stress = rec[0], rec[1], rec[2], rec[3], rec[4], rec[5]
            rec_exercise, rec_mood = rec[10], rec[11]
            rec_score = rec[9]
            preview = f"久坐{rec_sit} · 睡眠{rec_sleep}h · 运动{rec_exercise} · {rec_mood}"

            with st.expander(f"📅 {rec_date}  |  健康指数：{rec_score if rec_score else '未知'}  |  {preview}"):
                st.markdown(rec[8] if rec[8] else "（无详细数据）")
                col_del1, col_del2 = st.columns([3, 1])
                with col_del2:
                    if st.button(f"🗑 删除此记录", key=f"del_{i}"):
                        if delete_record(username, rec_date):
                            st.success("✅ 记录已删除")
                            st.rerun()
                        else:
                            st.error("❌ 删除失败")

    # 全部历史统计
    with st.expander("📊 查看全部历史数据统计"):
        all_records = get_user_records_all(username)
        if all_records:
            df = pd.DataFrame(all_records, columns=[
                "日期", "久坐", "低头", "睡眠(小时)", "饮水量", "压力评分",
                "天气", "气温", "AI回复", "综合健康指数", "运动时长", "心情"
            ])
            st.dataframe(df[["日期", "综合健康指数", "睡眠(小时)", "压力评分", "运动时长", "心情", "天气"]],
                         use_container_width=True, hide_index=True)
            # 添加趣味统计
            avg_all = get_average_score(username, 999)
            st.markdown(f"""
            <div style="display:flex; gap:16px; margin-top:12px; flex-wrap:wrap;">
                <div style="flex:1; min-width:120px; background:#f0f9ff; border-radius:12px; padding:12px; text-align:center;">
                    <div style="font-size:1.5rem;">📊</div>
                    <div style="font-size:0.8rem; color:#888;">全部平均分</div>
                    <div style="font-weight:700; font-size:1.2rem;">{avg_all if avg_all else 'N/A'}</div>
                </div>
                <div style="flex:1; min-width:120px; background:#f0fff4; border-radius:12px; padding:12px; text-align:center;">
                    <div style="font-size:1.5rem;">📅</div>
                    <div style="font-size:0.8rem; color:#888;">最早记录</div>
                    <div style="font-weight:700; font-size:1.2rem;">{all_records[-1][0] if len(all_records) > 0 else 'N/A'}</div>
                </div>
                <div style="flex:1; min-width:120px; background:#fff5f5; border-radius:12px; padding:12px; text-align:center;">
                    <div style="font-size:1.5rem;">🏆</div>
                    <div style="font-size:0.8rem; color:#888;">最高分</div>
                    <div style="font-weight:700; font-size:1.2rem;">{max(r[9] for r in all_records if r[9] is not None) if any(r[9] is not None for r in all_records) else 'N/A'}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("暂无数据")


if __name__ == "__main__":
    main()