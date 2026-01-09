# automation_config.py
SCHEDULE = {
    "daily_post_time": "09:00",  # Morning posting
    "weekly_digest_day": "Friday",
    "weekly_digest_time": "17:00",
    "cooldown_days": 3,
    "max_posts_per_day": 3,
}

CONTENT_TIPS = {
    "Instagram": {
        "optimal_length": 2200,
        "hashtags_min": 5,
        "hashtags_max": 30,
        "best_time": ["11:00", "13:00", "19:00"],
    },
    "Facebook": {
        "optimal_length": 250,
        "hashtags_min": 2,
        "hashtags_max": 5,
        "best_time": ["09:00", "13:00", "18:00"],
    },
    "LinkedIn": {
        "optimal_length": 1500,
        "hashtags_min": 3,
        "hashtags_max": 10,
        "best_time": ["08:00", "12:00", "17:00"],
    },
    "X": {
        "optimal_length": 280,
        "hashtags_min": 1,
        "hashtags_max": 3,
        "best_time": ["08:00", "12:00", "19:00"],
    },
}

# Compelling content formulas
CONTENT_FORMULAS = [
    "Problem + Solution + Property",
    "Statistic + Insight + Opportunity",
    "Story + Lesson + Investment",
    "Question + Answer + Call-to-Action",
    "Trend + Analysis + Recommendation",
]