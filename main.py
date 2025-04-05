import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Handle OpenMP errors

import sqlite3
import time
import logging
import joblib
import discord
from discord import ui, Button, Embed
from dotenv import load_dotenv
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
MODERATION_CHANNEL_ID = int(os.getenv("MODERATION_CHANNEL_ID"))  # Channel for mod review

distilbert_classifier = pipeline("text-classification", model="unitary/toxic-bert")
roberta_classifier = pipeline("text-classification", model="unitary/unbiased-toxic-roberta")

logging.info("Setting up SQLite database...")
conn = sqlite3.connect("moderation_data.db")
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS moderation (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    message TEXT,
    author TEXT,
    distilbert_score REAL,
    roberta_score REAL,
    logreg_score REAL,
    naive_bayes_score REAL,
    distilbert_latency REAL,
    roberta_latency REAL,
    logreg_latency REAL,
    naive_bayes_latency REAL,
    flagged_distilbert BOOLEAN,
    flagged_roberta BOOLEAN,
    flagged_logreg BOOLEAN,
    flagged_naive_bayes BOOLEAN,
    reviewed BOOLEAN DEFAULT 0,
    actual_label TEXT DEFAULT 'unknown',
    channel_id INTEGER,
    notification_id INTEGER DEFAULT NULL
)
""")
conn.commit()

logging.info("Loading trained models...")
try:
    vectorizer = joblib.load("vectorizer.pkl")
    logreg_model = joblib.load("logistic_regression_model.pkl")
    naive_bayes_model = joblib.load("naive_bayes_model.pkl")
    logging.info("✅ Successfully loaded ML models.")
except FileNotFoundError:
    logging.warning("⚠️ Trained models not found. Train them first!")
    logreg_model = None
    naive_bayes_model = None
    vectorizer = None

THRESHOLD_DISTILBERT = 0.75
THRESHOLD_ROBERTA = 0.75
THRESHOLD_LOGREG = 0.75
THRESHOLD_NAIVE_BAYES = 0.75

def detect_toxicity(model, message, threshold):
    start_time = time.time()
    result = model(message)
    latency = time.time() - start_time
    score = result[0]['score'] if isinstance(result, list) else 0
    flagged = score > threshold
    return score, flagged, latency

def detect_toxicity_logreg(message):
    if logreg_model and vectorizer:
        start_time = time.time()
        vectorized_message = vectorizer.transform([message])
        score = logreg_model.predict_proba(vectorized_message)[0, 1] 
        latency = time.time() - start_time
        flagged = score > THRESHOLD_LOGREG 
        return score, flagged, latency
    return 0, False, 0


def detect_toxicity_naive_bayes(message):
    if naive_bayes_model and vectorizer:
        start_time = time.time()
        vectorized_message = vectorizer.transform([message])
        probability = naive_bayes_model.predict_proba(vectorized_message)[0, 1]
        latency = time.time() - start_time
        flagged = probability > THRESHOLD_NAIVE_BAYES
        return probability, flagged, latency
    return 0, False, 0


def log_moderation_result(message, author, scores, latencies, flags, channel_id, notification_id=None):
    logging.info("Logging message to database...")
    
    values = [message, author, *scores, *latencies, *flags, 0, 'unknown', channel_id, notification_id]
    
    cursor.execute("""
    INSERT INTO moderation (message, author, distilbert_score, roberta_score, logreg_score, naive_bayes_score,
                            distilbert_latency, roberta_latency, logreg_latency, naive_bayes_latency,
                            flagged_distilbert, flagged_roberta, flagged_logreg, flagged_naive_bayes, reviewed, actual_label, channel_id, notification_id)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, values)
    conn.commit()

# Set up Discord bot
intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

@client.event
async def on_ready():
    logging.info(f"✅ Logged in as {client.user}")

class ModerationView(ui.View):
    def __init__(self, message_content, message_id, channel_id, notification_id):
        super().__init__(timeout=None)
        self.message_content = message_content
        self.message_id = message_id
        self.channel_id = channel_id
        self.notification_id = notification_id

    async def update_moderation_action(self, interaction: discord.Interaction, action: str):
        mod_channel = client.get_channel(MODERATION_CHANNEL_ID)
        if mod_channel:
            try:
                mod_message = await mod_channel.fetch_message(interaction.message.id)
                
                embed = mod_message.embeds[0]
                
                embed.set_field_at(
                    index=2,  
                    name="Moderation Action:",
                    value=f"Reviewed: {action}",
                    inline=False
                )
                
                await mod_message.edit(embed=embed)
            except discord.errors.NotFound:
                logging.error(f"Moderation message with ID {interaction.message.id} not found.")
            except IndexError:
                logging.error("Embed or field not found in the moderation message.")

    @ui.button(label="✅ Not Toxic", style=discord.ButtonStyle.success)
    async def not_toxic(self, interaction: discord.Interaction, button: Button):
        cursor.execute("UPDATE moderation SET reviewed = 1, actual_label = 'not_toxic' WHERE message = ?", (self.message_content,))
        conn.commit()
        
        try:
            channel = client.get_channel(self.channel_id)
            if channel:
                msg = await channel.fetch_message(self.message_id)
                
                await msg.clear_reaction("⚠️")
                
                notification_msg = await channel.fetch_message(self.notification_id)
                await notification_msg.delete()
                
                await self.update_moderation_action(interaction, "Not Toxic")
                
                await interaction.response.send_message("Message marked as not toxic. Reaction and notification removed.", ephemeral=True)
            else:
                await interaction.response.send_message("Error: The channel could not be found.", ephemeral=True)
        except discord.errors.NotFound:
            await interaction.response.send_message("Error: The message or notification no longer exists.", ephemeral=True)
            logging.error(f"Message or notification with ID {self.message_id} or {self.notification_id} not found.")

    @ui.button(label="❌ Toxic", style=discord.ButtonStyle.danger)
    async def toxic(self, interaction: discord.Interaction, button: Button):
        cursor.execute("UPDATE moderation SET reviewed = 1, actual_label = 'toxic' WHERE message = ?", (self.message_content,))
        conn.commit()
        
        try:
            channel = client.get_channel(self.channel_id)
            if channel:
                msg = await channel.fetch_message(self.message_id)
                
                await self.update_moderation_action(interaction, "Toxic")
                
                await interaction.response.send_message("Message marked as toxic.", ephemeral=True)
            else:
                await interaction.response.send_message("Error: The channel could not be found.", ephemeral=True)
        except discord.errors.NotFound:
            await interaction.response.send_message("Error: The message no longer exists.", ephemeral=True)
            logging.error(f"Message with ID {self.message_id} not found.")

@client.event
async def on_message(message):
    if message.author == client.user:
        return
    
    distilbert_score, flagged_distilbert, distilbert_latency = detect_toxicity(distilbert_classifier, message.content, THRESHOLD_DISTILBERT)
    roberta_score, flagged_roberta, roberta_latency = detect_toxicity(roberta_classifier, message.content, THRESHOLD_ROBERTA)
    logreg_score, flagged_logreg, logreg_latency = detect_toxicity_logreg(message.content)
    naive_bayes_score, flagged_naive_bayes, naive_bayes_latency = detect_toxicity_naive_bayes(message.content)
    
    scores = [distilbert_score, roberta_score, logreg_score, naive_bayes_score]
    latencies = [distilbert_latency, roberta_latency, logreg_latency, naive_bayes_latency]
    flags = [flagged_distilbert, flagged_roberta, flagged_logreg, flagged_naive_bayes]
    
    notification_id = None
    if sum(flags) >= 2:
        await message.add_reaction("⚠️")  # Add a warning emoji
        
        notification = await message.channel.send(f"{message.author.mention}, your message has been flagged for potential toxicity. It is under review by moderators.")
        notification_id = notification.id
        
        mod_channel = client.get_channel(MODERATION_CHANNEL_ID)
        if mod_channel:
            embed = Embed(
                title="⚠️ Potential Toxic Message",
                description=f"Message from **{message.author.name}** in <#{message.channel.id}>",
                color=discord.Color.orange()
            )
            embed.add_field(name="Message:", value=f"```{message.content}```", inline=False)
            embed.add_field(name="Scores:", value=f"DistilBERT: {distilbert_score}\nRoBERTa: {roberta_score}\nLogReg: {logreg_score}\nNaive Bayes: {naive_bayes_score}", inline=False)
            embed.add_field(name="Moderation Action:", value="Pending review", inline=False)
            
            await mod_channel.send(embed=embed, view=ModerationView(message.content, message.id, message.channel.id, notification_id))
    
    log_moderation_result(message.content, message.author.name, scores, latencies, flags, message.channel.id, notification_id)

client.run(DISCORD_BOT_TOKEN)
