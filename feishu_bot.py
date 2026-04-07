import json
import requests
import re
from flask import Flask, request, jsonify
import os

# 设置 HuggingFace 镜像（解决网络问题）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "0"

# 导入你的 AI 问答函数
from app import get_ai_answer
processed_events = set()
app = Flask(__name__)

# ========== 飞书凭证 ==========
APP_ID = "cli_a944f5c769b81cbb"
APP_SECRET = "oZ6LLGZUugnuiYjx4ciTBddYQSHSMrGw"  #


# ========== 获取 tenant_access_token ==========
def get_tenant_access_token():
    url = "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal"
    headers = {"Content-Type": "application/json"}
    data = {"app_id": APP_ID, "app_secret": APP_SECRET}
    resp = requests.post(url, headers=headers, json=data)
    return resp.json().get("tenant_access_token")


# ========== 发送消息到群聊 ==========
def send_message_to_chat(chat_id, text):
    token = get_tenant_access_token()
    # 1. 将 receive_id_type 作为查询参数
    url = f"https://open.feishu.cn/open-apis/im/v1/messages?receive_id_type=chat_id"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    # 2. 请求体中只包含三个字段
    data = {
        "receive_id": chat_id,
        "msg_type": "text",
        "content": json.dumps({"text": text})
    }
    print("【发送请求的完整URL】", url)
    print("【发送请求的Headers】", headers)
    print("【发送请求的Body】", data)
    try:
        resp = requests.post(url, headers=headers, json=data)
        print("【响应状态码】", resp.status_code)
        print("【响应内容】", resp.text)
        if resp.status_code == 200:
            print("消息发送成功")
            return True
        else:
            print("消息发送失败")
            return False
    except Exception as e:
        print(f"请求发生异常: {e}")
        return False
# ========== 处理飞书 Webhook ==========
@app.route("/webhook", methods=["POST"])
def webhook():
    data = request.get_json()
    print("=== 收到飞书请求 ===")
    print(json.dumps(data, indent=2, ensure_ascii=False))

    # 幂等性：检查事件是否已处理
    event_id = data.get("header", {}).get("event_id")
    if event_id in processed_events:
        print(f"重复事件 {event_id}，忽略")
        return "success"
    processed_events.add(event_id)

    # 可选：限制集合大小，避免内存无限增长
    if len(processed_events) > 1000:
        processed_events.clear()

    # 验证回调
    if data.get("challenge"):
        return jsonify({"challenge": data["challenge"]})

    event = data.get("event", {})
    message = event.get("message", {})

    msg_type = message.get("message_type")
    if msg_type != "text":
        print(f"忽略非文本消息，类型: {msg_type}")
        return "success"

    # 提取用户消息文本（处理多种格式）
    content_raw = message.get("content", "")
    user_text = ""
    if content_raw:
        try:
            # 尝试解析 JSON
            content_json = json.loads(content_raw)
            user_text = content_json.get("text", "")
        except:
            # 用正则提取
            match = re.search(r'"text":"([^"]*)"', content_raw)
            user_text = match.group(1) if match else ""

    if user_text and " " in user_text:
        user_text = user_text.split(" ", 1)[1]
    else:
        if not user_text.strip():
            user_text = "你好"  # 默认给一个问候

    chat_id = message.get("chat_id")
    if not chat_id:
        print("未获取到 chat_id")
        return "success"

    print(f"用户消息: {user_text}, chat_id: {chat_id}")

    # 调用你的 AI 函数获取回复
    try:
        reply = get_ai_answer(user_text)
    except Exception as e:
        print(f"AI 调用失败: {e}")
        reply = "AI 服务暂时不可用，请稍后再试。"

    # 发送回复
    success = send_message_to_chat(chat_id, reply)
    if success:
        print("回复发送成功")
    else:
        print("回复发送失败")

    return "success"


if __name__ == "__main__":
    app.run(port=8000, debug=False)