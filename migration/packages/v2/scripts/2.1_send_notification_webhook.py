import os

import pymsteams

myTeamsMessage = pymsteams.connectorcard(os.getenv("MS_TEAMS_CHANNEL"))
myTeamsMessage.text("Training process finished")
myTeamsMessage.send()

last_status_code = myTeamsMessage.last_http_response.status_code
