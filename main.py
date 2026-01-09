from dotenv import load_dotenv
import os
import smtplib
import ssl
from email.message import EmailMessage
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta
from random import choice
import requests
import feedparser

load_dotenv()

LIVE_POSTING = os.getenv("LIVE_POSTING", "false").lower() == "true"

EMAIL_SENDER = os.getenv("EMAIL_SENDER")
EMAIL_APP_PASSWORD = os.getenv("EMAIL_APP_PASSWORD")
EMAIL_RECIPIENT = os.getenv("EMAIL_RECIPIENT")
EMAIL_NOTIFICATION = os.getenv("EMAIL_NOTIFICATION", "true").lower() == "true"

META_ACCESS_TOKEN = os.getenv("META_ACCESS_TOKEN")
META_PAGE_ID = os.getenv("META_PAGE_ID")

X_BEARER_TOKEN = os.getenv("X_BEARER_TOKEN")

WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN")
WHATSAPP_PHONE_ID = os.getenv("WHATSAPP_PHONE_ID")

PROJECT_ROOT = Path(__file__).parent.resolve()
EXCEL_PATH = PROJECT_ROOT / "data" / "Product_Information_Base.xlsx"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

PLATFORMS = ["Instagram", "Facebook", "LinkedIn", "X", "GoogleMyBusiness"]

SELLING_STRATEGIES = [
    "EMOTIONAL", "FOMO", "SOCIAL_PROOF", "AUTHORITY", "NEWS_SOLUTION", "HUMOUR"
]

UNDERUSED_CITIES = ["Abuja", "Port Harcourt", "Ibadan", "Asaba", "Uyo"]

NEWS_FEED = "https://news.google.com/rss/search?q=Nigerian+real+estate+investment"

# -------------------------------------------------
# CORE LOADERS
# -------------------------------------------------
def load_products():
    df = pd.read_excel(EXCEL_PATH, sheet_name="Products")
    df.columns = df.columns.str.strip().str.lower()
    df = df[df["content_status"].str.lower() == "active"]
    
    # Ensure proper dtypes for date columns
    if "last_content_date" in df.columns:
        df["last_content_date"] = pd.to_datetime(df["last_content_date"], errors="coerce")
    
    # Fill NaN values in text columns with empty string
    text_columns = ["product_name", "location", "product_desc"]
    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)
    
    return df


def load_calendar():
    try:
        df = pd.read_excel(EXCEL_PATH, sheet_name="Content_Calendar")
        df.columns = df.columns.str.strip().str.lower()
        
        # Ensure date column is datetime
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
            
        return df
    except Exception:
        return pd.DataFrame(columns=[
            "date", "post_id", "product_id", "platform",
            "strategy", "engagement", "variant"
        ])


# -------------------------------------------------
# INTELLIGENCE LAYERS
# -------------------------------------------------
def apply_cooldown(df, days=3):
    today = pd.Timestamp.now().normalize()
    
    # Ensure last_content_date is datetime
    if "last_content_date" not in df.columns:
        return df
    
    df["last_content_date"] = pd.to_datetime(df["last_content_date"], errors="coerce")
    
    return df[
        df["last_content_date"].isna() |
        ((today - df["last_content_date"]).dt.days >= days)
    ]


def strategy_weights(calendar_df):
    if calendar_df.empty:
        return {s: 1 for s in SELLING_STRATEGIES}
    
    scores = calendar_df.groupby("strategy")["engagement"].sum().to_dict()
    return {s: max(scores.get(s, 1), 1) for s in SELLING_STRATEGIES}


def pick_strategy(product_id, calendar_df):
    weights = strategy_weights(calendar_df)
    pool = []
    for k, v in weights.items():
        pool.extend([k] * v)
    
    recent = calendar_df[calendar_df["product_id"] == product_id]["strategy"].tail(2).tolist()
    filtered = [s for s in pool if s not in recent]
    
    return choice(filtered) if filtered else choice(pool)


# -------------------------------------------------
# NEWS → PRODUCT MATCHING
# -------------------------------------------------
def fetch_news():
    feed = feedparser.parse(NEWS_FEED)
    return [e.title for e in feed.entries[:5]] if feed.entries else []


def match_news_to_product(news, product):
    # Safely handle product_desc which might be NaN
    product_desc = product.get("product_desc", "")
    if pd.isna(product_desc) or not isinstance(product_desc, str):
        return False
    
    keywords = product_desc.lower()
    return any(word in keywords for word in news.lower().split())


# -------------------------------------------------
# CONTENT GENERATION - ENHANCED
# -------------------------------------------------
def generate_image_prompt(product):
    location = product.get("location", "Nigeria")
    return f"High quality Nigerian real estate image, {location}, modern investment property, daylight, professional"


def generate_story_content(product):
    """Generate narrative-driven content - ADVANCED CONTENT IDEA #1"""
    stories = [
        f"Meet Ade, a UK-based nurse who bought land in {product.get('location', 'Nigeria')} in 2020. "
        f"Today, her {product.get('product_name', 'property')} plot has tripled in value. "
        f"Her story could be yours...",
        
        f"From saving ₦50k monthly to owning prime property. "
        f"How {product.get('product_name', 'this property')} helped a teacher build wealth "
        f"while working abroad. The blueprint is simple...",
        
        f"The diaspora dream: A home back home. "
        f"{product.get('product_name', 'This property')} in {product.get('location', 'Nigeria')} makes it reality. "
        f"What's stopping you from starting?"
    ]
    return choice(stories)


def generate_urgency_content(product):
    """Create time-sensitive content - ADVANCED CONTENT IDEA #2"""
    urgency_triggers = [
        f"Price review this Friday for {product.get('product_name', 'this property')}",
        f"Only 2 units left at launch price for {product.get('product_name', 'this development')}",
        f"Early bird discount ends tonight for {product.get('location', 'this location')} plots",
        f"Market analysis suggests 15% price increase next month for {product.get('location', 'this area')}"
    ]
    return choice(urgency_triggers)


def generate_benefit_content(product):
    """Focus on benefits over features - ADVANCED CONTENT IDEA #3"""
    benefits = [
        "significant tax advantages",
        "powerful inflation hedge", 
        "consistent passive income potential",
        "secure generational wealth creation",
        "smart portfolio diversification"
    ]
    return f"{product.get('product_name', 'This property')} offers {choice(benefits)} in {product.get('location', 'Nigeria')}. " \
           f"Learn how it works..."


def generate_compelling_content(product, platform, strategy, variant, news=None):
    """Generate more compelling content with AI-like creativity - SECTION 2"""
    
    name = product.get("product_name", "Property")
    location = product.get("location", "Nigeria")
    price = product.get("price", "Contact for price")
    desc = product.get("product_desc", "A premium real estate investment opportunity.")
    
    # Convert price to more readable format
    try:
        if isinstance(price, (int, float)):
            price_formatted = f"₦{price:,.0f}"
        elif "₦" in str(price) or "NGN" in str(price).upper():
            price_formatted = str(price)
        else:
            price_formatted = f"₦{price}"
    except:
        price_formatted = price
    
    # Enhanced hooks for different strategies
    hooks = {
        "EMOTIONAL": [
            f"Imagine building generational wealth from {location}. {name} makes this possible.",
            f"This isn't just property, it's a legacy. {name} in {location} awaits.",
            f"Your family's future home starts here. Discover {name} in {location}."
        ],
        "FOMO": [
            f"⚠️ Limited units available! {name} in {location} is 60% sold out.",
            f"Last 3 plots remaining at {name}. Investors are securing theirs now.",
            f"Price increases next week. Lock in today's rate at {name}, {location}."
        ],
        "SOCIAL_PROOF": [
            f"Join 47 other smart investors who chose {name} in {location} this month.",
            f"Our fastest-selling property: {name}. See why investors love {location}.",
            f"Featured in 'Top 10 Investment Properties': {name}, {location}."
        ],
        "AUTHORITY": [
            f"Market analysis shows {location} will appreciate 25% in 2024. {name} leads the way.",
            f"Expert tip: Properties like {name} in {location} hedge against inflation.",
            f"According to real estate analysts, {location} is undervalued. {name} represents opportunity."
        ],
        "NEWS_SOLUTION": [
            f"{news if news else 'Recent market developments'} make {name} in {location} a strategic buy.",
            f"While others worry about market news, smart investors buy {name} in {location}.",
            f"Market shift alert: {news if news else 'Industry changes'} favor properties like {name}."
        ],
        "HUMOUR": [
            f"Your money in the bank: 😴 | Your money in {name}: 🚀",
            f"They said 'buy land, they're not making more.' We found more at {name}!",
            f"Stock market: 📉 | {name} in {location}: 📈 (Your move)"
        ]
    }
    
    # Select random hook from strategy category
    hook = choice(hooks.get(strategy, [f"{name} in {location} is gaining attention."]))
    
    # Add advanced content ideas based on strategy
    content_extensions = []
    if strategy == "EMOTIONAL":
        content_extensions.append(generate_story_content(product))
    elif strategy == "FOMO":
        content_extensions.append(generate_urgency_content(product))
    elif strategy == "AUTHORITY":
        content_extensions.append(generate_benefit_content(product))
    
    # Platform-specific content
    if platform == "X":
        hashtags = "#NigeriaRealEstate #PropertyInvestment #Diaspora #RealEstate #Investment"
        if "Lagos" in location:
            hashtags += " #LagosRealEstate"
        elif "Abuja" in location:
            hashtags += " #AbujaProperties"
        
        # X has character limit - keep it tight
        full_content = f"{hook}"
        if content_extensions:
            full_content += f"\n\n{content_extensions[0]}"
        full_content += f"\n\n{price_formatted}\n\n{hashtags}"
        
        # Truncate if too long for X
        if len(full_content) > 280:
            full_content = full_content[:275] + "..."
        return full_content
    
    elif platform == "Instagram":
        hashtags = "#NigeriaRealEstate #PropertyInvestment #DiasporaInvestors #RealEstateNigeria #LuxuryLiving"
        if content_extensions:
            caption = f"{hook}\n\n{content_extensions[0]}\n\n{desc}\n\n💵 From {price_formatted}\n📍 {location}\n📱 WhatsApp +2348024427735\n\n{hashtags}"
        else:
            caption = f"{hook}\n\n{desc}\n\n💵 From {price_formatted}\n📍 {location}\n📱 WhatsApp +2348024427735\n\n{hashtags}"
        return caption
    
    elif platform == "LinkedIn":
        hashtags = "#RealEstateInvestment #PropertyDevelopment #NigeriaBusiness #DiasporaInvestment #WealthBuilding"
        if content_extensions:
            caption = f"{hook}\n\n{content_extensions[0]}\n\n{desc}\n\n🔑 Investment Opportunity: {price_formatted}\n📍 Location: {location}\n\nFor serious investors only. DM for portfolio review.\n\n{hashtags}"
        else:
            caption = f"{hook}\n\n{desc}\n\n🔑 Investment Opportunity: {price_formatted}\n📍 Location: {location}\n\nFor serious investors only. DM for portfolio review.\n\n{hashtags}"
        return caption
    
    # Default for Facebook/GoogleMyBusiness
    hashtags = "#RealEstate #Property #Investment #Nigeria #Diaspora"
    if content_extensions:
        return f"{hook}\n\n{content_extensions[0]}\n\n{desc}\n\n💰 Price: {price_formatted}\n📍 Location: {location}\n📞 Contact: +2348024427735\n\n{hashtags}"
    return f"{hook}\n\n{desc}\n\n💰 Price: {price_formatted}\n📍 Location: {location}\n📞 Contact: +2348024427735\n\n{hashtags}"


# -------------------------------------------------
# ENGAGEMENT INGESTION (META + X)
# -------------------------------------------------
def pull_meta_engagement(post_id):
    return 0


def pull_x_engagement(post_id):
    return 0


# -------------------------------------------------
# WHATSAPP BROADCAST
# -------------------------------------------------
def send_whatsapp(message):
    if not LIVE_POSTING:
        return
    
    try:
        requests.post(
            f"https://graph.facebook.com/v19.0/{WHATSAPP_PHONE_ID}/messages",
            headers={"Authorization": f"Bearer {WHATSAPP_TOKEN}"},
            json={
                "messaging_product": "whatsapp",
                "to": EMAIL_RECIPIENT,
                "type": "text",
                "text": {"body": message}
            }
        )
    except Exception as e:
        print(f"WhatsApp sending failed: {e}")


# -------------------------------------------------
# AUTHORITY DIGEST
# -------------------------------------------------
def generate_authority_digest(calendar_df):
    week = calendar_df.tail(20)
    html = "<h1>Weekly Real Estate Authority Digest</h1>"
    
    for _, row in week.iterrows():
        html += f"<h3>{row['platform']} | {row['strategy']}</h3><p>{row['post_id']}</p>"
    
    path = OUTPUT_DIR / f"digest_{datetime.now().date()}.html"
    path.write_text(html, encoding="utf-8")
    return path


# -------------------------------------------------
# HTML EMAIL TEMPLATES - SECTION 1
# -------------------------------------------------
def generate_html_email(daily_log, products_df, calendar_df):
    """Generate beautiful HTML email with analytics"""
    
    # Calculate simple metrics
    total_posts = len(daily_log)
    platforms_used = []
    for post in daily_log:
        if "]" in post:
            platform = post.split("]")[0].replace("[", "")
            if platform not in platforms_used:
                platforms_used.append(platform)
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; max-width: 800px; margin: 0 auto; padding: 20px; }}
            .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 10px 10px 0 0; }}
            .content {{ background: #f9f9f9; padding: 30px; border-radius: 0 0 10px 10px; }}
            .metric-box {{ background: white; padding: 20px; border-radius: 8px; margin: 15px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            .post-card {{ background: white; padding: 20px; margin: 15px 0; border-radius: 8px; border-left: 5px solid #667eea; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            .platform-badge {{ display: inline-block; background: #667eea; color: white; padding: 5px 15px; border-radius: 20px; font-size: 12px; margin-right: 10px; }}
            .strategy-badge {{ display: inline-block; background: #764ba2; color: white; padding: 5px 15px; border-radius: 20px; font-size: 12px; }}
            h1 {{ margin: 0; }}
            h2 {{ color: #667eea; border-bottom: 2px solid #f0f0f0; padding-bottom: 10px; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🏠 Real Estate Authority Daily Report</h1>
            <p>{datetime.now().strftime('%A, %B %d, %Y')}</p>
        </div>
        
        <div class="content">
            <div class="metric-box">
                <h2>📊 Daily Summary</h2>
                <p><strong>Total Posts Generated:</strong> {total_posts}</p>
                <p><strong>Platforms Targeted:</strong> {', '.join(platforms_used) if platforms_used else 'None'}</p>
                <p><strong>Active Products in Database:</strong> {len(products_df)}</p>
            </div>
            
            <h2>📝 Generated Content</h2>
    """
    
    # Add each post as a card
    for i, post in enumerate(daily_log, 1):
        # Extract platform from log
        platform = post.split("]")[0].replace("[", "") if "]" in post else "Unknown"
        content = post.split("] ")[1] if "] " in post else post
        
        html += f"""
            <div class="post-card">
                <div style="margin-bottom: 15px;">
                    <span class="platform-badge">{platform}</span>
                    <span class="strategy-badge">Content #{i}</span>
                </div>
                <p style="white-space: pre-wrap;">{content}</p>
            </div>
        """
    
    # Add footer
    html += f"""
            <div style="margin-top: 30px; padding-top: 20px; border-top: 1px solid #eee; color: #666; font-size: 12px;">
                <p><em>This is an automated report from your Real Estate Authority AI System.</em></p>
                <p>System generated at: {datetime.now().strftime('%I:%M %p')}</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    return html


# -------------------------------------------------
# CITY UNDERUSE FLAGGING
# -------------------------------------------------
def flag_underused_cities(products_df):
    if "location" not in products_df.columns:
        return UNDERUSED_CITIES
    
    used = products_df["location"].value_counts().to_dict()
    return [c for c in UNDERUSED_CITIES if used.get(c, 0) < 2]


# -------------------------------------------------
# MAIN - ENHANCED (SECTION 2)
# -------------------------------------------------
def main():
    print(f"🚀 Starting Real Estate Authority Automation - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    
    # Load data
    products_df = load_products()
    calendar_df = load_calendar()
    news_items = fetch_news()
    
    print(f"📊 Loaded {len(products_df)} active products, {len(calendar_df)} calendar entries")
    
    # Check for underused cities
    underused = flag_underused_cities(products_df)
    if underused:
        print(f"📍 Underused cities flagged: {', '.join(underused)}")
    
    # Apply cooldown and select products
    eligible = apply_cooldown(products_df)
    daily_log = []
    
    sample_size = min(3, len(eligible))
    if sample_size == 0:
        print("❌ No eligible products found. Check cooldown period or activate more products.")
        return
    
    print(f"🎯 Selected {sample_size} products for today's content")
    
    # Generate content for selected products
    for idx, (_, product) in enumerate(eligible.sample(sample_size).iterrows()):
        print(f"\n📦 Processing: {product.get('product_name', 'Unknown Product')}")
        
        strategy = pick_strategy(product["product_id"], calendar_df)
        print(f"   Strategy: {strategy}")
        
        for variant in ["A", "B"]:
            for platform in PLATFORMS:
                # Match news if available
                matched_news = None
                for news in news_items:
                    if match_news_to_product(news, product):
                        matched_news = news
                        break
                
                # Generate compelling content
                text = generate_compelling_content(product, platform, strategy, variant, matched_news)
                
                # Add to calendar
                new_row = {
                    "date": datetime.now().date(),
                    "post_id": f"{platform}-{product['product_id']}-{variant}-{datetime.now().strftime('%Y%m%d')}",
                    "product_id": product["product_id"],
                    "platform": platform,
                    "strategy": strategy,
                    "engagement": 0,
                    "variant": variant,
                    "content": text[:200] + "..." if len(text) > 200 else text  # Store excerpt
                }
                
                calendar_df = pd.concat([calendar_df, pd.DataFrame([new_row])], ignore_index=True)
                daily_log.append(f"[{platform}] {text}")
        
        # Update product metrics
        if "usage_count" in products_df.columns:
            current_count = products_df.loc[products_df["product_id"] == product["product_id"], "usage_count"].values
            if len(current_count) > 0:
                products_df.loc[products_df["product_id"] == product["product_id"], "usage_count"] = current_count[0] + 1
        
        if "last_content_date" in products_df.columns:
            products_df.loc[products_df["product_id"] == product["product_id"], "last_content_date"] = pd.Timestamp.now().normalize()
    
    # Save updated data
    try:
        with pd.ExcelWriter(EXCEL_PATH, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
            products_df.to_excel(writer, sheet_name="Products", index=False)
            calendar_df.to_excel(writer, sheet_name="Content_Calendar", index=False)
        print(f"💾 Data saved to {EXCEL_PATH}")
    except Exception as e:
        print(f"❌ Error saving to Excel: {e}")
        return
    
    # Generate and save digest
    try:
        digest = generate_authority_digest(calendar_df)
        print(f"📄 Digest saved to: {digest}")
    except Exception as e:
        print(f"⚠️ Error generating digest: {e}")
    
    # Send WhatsApp message if LIVE_POSTING
    if LIVE_POSTING:
        try:
            whatsapp_msg = "🏡 New real estate investment opportunities just dropped!\n\n" \
                         f"We've analyzed {len(products_df)} properties and generated fresh content.\n" \
                         "Reply 'INFO' for detailed listings or 'CALL' to speak with an agent."
            send_whatsapp(whatsapp_msg)
            print("📱 WhatsApp message sent")
        except Exception as e:
            print(f"⚠️ Error sending WhatsApp: {e}")
    else:
        print("📱 WhatsApp: Dry run (LIVE_POSTING=false)")
    
    # Send HTML email notification
    if EMAIL_NOTIFICATION and daily_log:
        try:
            msg = EmailMessage()
            msg["From"] = EMAIL_SENDER
            msg["To"] = EMAIL_RECIPIENT
            msg["Subject"] = f"🏠 Real Estate Daily Report - {datetime.now().strftime('%B %d, %Y')}"
            
            # Generate HTML content
            html_content = generate_html_email(daily_log, products_df, calendar_df)
            
            # Set both plain text and HTML versions
            plain_text = "DAILY REAL ESTATE AUTHORITY REPORT\n" + "="*50 + "\n\n"
            for p in daily_log:
                platform = p.split("]")[0].replace("[", "") if "]" in p else "Unknown"
                content = p.split("] ")[1] if "] " in p else p
                plain_text += f"--- {platform} ---\n{content}\n\n"
            
            msg.set_content(plain_text)
            msg.add_alternative(html_content, subtype='html')
            
            with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=ssl.create_default_context()) as server:
                server.login(EMAIL_SENDER, EMAIL_APP_PASSWORD)
                server.send_message(msg)
            print("📧 HTML email notification sent successfully")
        except Exception as e:
            print(f"❌ Error sending email: {e}")
    
    print(f"\n✅ Automation completed at {datetime.now().strftime('%H:%M:%S')}")
    print(f"📈 Generated {len(daily_log)} posts across {len(set([p.split(']')[0].replace('[', '') for p in daily_log if ']' in p]))} platforms")


if __name__ == "__main__":
    main()