import os
import smtplib
import sys
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

def load_env(env_path=".env"):
    """Manually load environment variables from a local .env file."""
    path = Path(env_path)
    if path.exists():
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, val = line.split("=", 1)
                    key = key.strip()
                    val = val.strip().strip('"').strip("'")
                    os.environ[key] = val
        except Exception as e:
            print(f"[Mailer] Error reading .env file: {e}", file=sys.stderr)

# Load environment variables on import
load_env()

def send_reset_email(to_email: str, username: str, token: str, app_url: str = "http://localhost:8501") -> tuple[bool, str]:
    """Send a password reset email using SMTP, with a terminal console fallback.
    
    Returns (success, message).
    """
    reset_link = f"{app_url}?token={token}"
    
    # Retrieve SMTP configuration
    smtp_host = os.environ.get("SMTP_HOST")
    smtp_port_str = os.environ.get("SMTP_PORT")
    smtp_user = os.environ.get("SMTP_USER")
    smtp_password = os.environ.get("SMTP_PASSWORD")
    smtp_from = os.environ.get("SMTP_FROM", smtp_user)
    
    # Check if SMTP is fully configured
    smtp_configured = all([smtp_host, smtp_port_str, smtp_user, smtp_password])
    
    # Console logging helper for developer fallback
    def log_fallback_to_console():
        border = "=" * 80
        console_msg = (
            f"\n{border}\n"
            f"DEVELOPER FALLBACK: PASSWORD RESET REQUEST\n"
            f"{border}\n"
            f"User:       {username}\n"
            f"Email:      {to_email}\n"
            f"Token:      {token}\n"
            f"Reset Link: {reset_link}\n"
            f"{border}\n"
        )
        # Print directly to stdout/stderr so it is visible in the terminal
        print(console_msg, flush=True)
        sys.stdout.write(console_msg)
        sys.stdout.flush()

    if not smtp_configured:
        log_fallback_to_console()
        return False, "SMTP environment variables are missing. Reset token and link logged to terminal console."

    try:
        smtp_port = int(smtp_port_str)
    except ValueError:
        log_fallback_to_console()
        return False, f"Invalid SMTP_PORT: '{smtp_port_str}'. Reset token and link logged to terminal console."

    # Construct the email
    msg = MIMEMultipart("alternative")
    msg["Subject"] = "Reset Your N-Deavour Password"
    msg["From"] = smtp_from
    msg["To"] = to_email

    # Plain text body
    text_body = (
        f"Hello {username},\n\n"
        f"We received a request to reset your N-Deavour password. Please use the link below to set a new password. "
        f"This link is valid for 15 minutes.\n\n"
        f"Reset Link: {reset_link}\n\n"
        f"If you did not request this reset, please ignore this email.\n\n"
        f"Best regards,\n"
        f"N-Deavour Alignment Team"
    )

    # HTML body with N-Deavour brand colors (Cloud Light #F8FAFC, Deep Teal #0D7A87)
    html_body = f"""
    <html>
      <body style="font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #F8FAFC; color: #0F172A; padding: 2rem; margin: 0;">
        <div style="max-width: 600px; margin: 0 auto; background-color: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; padding: 2.5rem; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);">
          <div style="text-align: center; margin-bottom: 2rem;">
            <h2 style="color: #0D7A87; margin: 0; font-size: 1.5rem; letter-spacing: -0.03em;">N-DEAVOUR ALIGNMENT</h2>
            <p style="color: #64748B; font-size: 0.85rem; margin: 0.25rem 0 0 0; text-transform: uppercase; letter-spacing: 0.05em;">Excellence through efficiency</p>
          </div>
          
          <p style="font-size: 1rem; line-height: 1.6; color: #0F172A;">Hello <strong style="color: #0F172A;">{username}</strong>,</p>
          
          <p style="font-size: 1rem; line-height: 1.6; color: #0F172A;">We received a request to reset your password. Click the button below to set a new password. This link is only valid for <strong>15 minutes</strong>.</p>
          
          <div style="text-align: center; margin: 2.5rem 0;">
            <a href="{reset_link}" style="background-color: #0D7A87; color: #FFFFFF; text-decoration: none; padding: 0.8rem 2rem; font-weight: bold; border-radius: 6px; display: inline-block; box-shadow: 0 4px 12px rgba(13, 122, 135, 0.25); transition: background-color 0.2s ease;">Reset Password</a>
          </div>
          
          <p style="font-size: 0.9rem; line-height: 1.6; color: #64748B;">If the button above does not work, copy and paste this URL into your browser:</p>
          <p style="font-size: 0.85rem; line-height: 1.5; color: #0F172A; word-break: break-all; background-color: #F1F5F9; padding: 0.75rem; border-radius: 4px;">{reset_link}</p>
          
          <hr style="border: 0; border-top: 1px solid #E2E8F0; margin: 2rem 0;" />
          
          <p style="font-size: 0.8rem; line-height: 1.5; color: #64748B; margin: 0;">If you did not request a password reset, you can safely ignore this email. Your password will remain unchanged.</p>
        </div>
      </body>
    </html>
    """

    msg.attach(MIMEText(text_body, "plain"))
    msg.attach(MIMEText(html_body, "html"))

    try:
        # Determine connection type based on port
        if smtp_port == 465:
            server = smtplib.SMTP_SSL(smtp_host, smtp_port, timeout=10)
        else:
            server = smtplib.SMTP(smtp_host, smtp_port, timeout=10)
            server.ehlo()
            if server.has_extn("STARTTLS"):
                server.starttls()
                server.ehlo()
                
        server.login(smtp_user, smtp_password)
        server.sendmail(smtp_from, [to_email], msg.as_string())
        server.quit()
        return True, "Password reset email sent successfully."
    except Exception as e:
        # Fall back to console logging if sending fails
        log_fallback_to_console()
        return False, f"SMTP delivery failed: {e}. Reset token and link logged to terminal console."
