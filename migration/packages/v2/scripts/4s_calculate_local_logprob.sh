source .env
uv run ./4_calculate_local_logprob.py --exp ${EXPERIMENT_NAME}
# uv run ./2.1_send_notification_webhook.py