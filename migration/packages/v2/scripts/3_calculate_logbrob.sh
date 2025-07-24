source .env
uv run ./3_calculate_logprob_test_dataset.py --exp ${EXPERIMENT_NAME}
# uv run ./2.1_send_notification_webhook.py