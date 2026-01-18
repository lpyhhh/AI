import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import Header

def send_email(subject, body, to_email, from_email, smtp_server, smtp_port, smtp_username, smtp_password):
    # 创建邮件对象
    msg = MIMEMultipart()
    msg['From'] = from_email
    msg['To'] = to_email
    msg['Subject'] = Header(subject, 'utf-8').encode()  # 使用 UTF-8 编码主题

    # 添加邮件正文（指定 UTF-8 编码）
    msg.attach(MIMEText(body, 'plain', 'utf-8'))

    try:
        # 连接到 SMTP 服务器
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()  # 启用 TLS 加密
        server.login(smtp_username, smtp_password)

        # 发送邮件
        server.sendmail(from_email, to_email, msg.as_string())
        print("邮件发送成功！")
        server.quit()
    except Exception as e:
        print(f"邮件发送失败: {e}")

if __name__ == "__main__":
    # 示例配置
    subject = "程序运行完成通知"
    body = "AI模型预测完成！"
    to_email = "496276044@qq.com"
    from_email = "496276044@qq.com"
    smtp_server = "smtp.qq.com"
    smtp_port = 587
    smtp_username = "496276044@qq.com"
    smtp_password = "ktujjaklwlhecbci"  # 替换为你的实际授权码

    send_email(subject, body, to_email, from_email, smtp_server, smtp_port, smtp_username, smtp_password)