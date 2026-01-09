# scheduler.py
import schedule
import time
from datetime import datetime

def job():
    print(f"Running automation at {datetime.now()}")
    from main import main
    main()

# Schedule daily at 9 AM
schedule.every().day.at("09:00").do(job)

print("🤖 Real Estate Authority Scheduler Started")
print("📅 Next run: Daily at 09:00 AM")

while True:
    schedule.run_pending()
    time.sleep(60)