source .env
uv run ./2_test.py --exp '${EXPERIMENT_NAME}'
# uv run ./2.1_send_notification_webhook.py