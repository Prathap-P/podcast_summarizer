import json
import random
from datetime import datetime, timezone

from flask import Flask, render_template, request, jsonify, send_file, stream_with_context, Response
import os
import shutil
from pathlib import Path

from langchain.chains.conversation.base import ConversationChain
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
import requests
import time

from dotenv import load_dotenv
load_dotenv()

from youtube_transcript_fetcher import get_youtube_transcript, extract_video_id, purge_expired_translation_cache
from audio_config import ASR_BACKEND, TTS_BACKEND
from system_prompts import subject_matter_expert_prompt
from condenser_service import (
    build_intro_prompt,
    condense_content,
    get_condensation_profile_id,
)
from condensation_cache import (
    compute_cache_key,
    load_checkpoint,
    create_checkpoint,
    save_checkpoint,
    get_progress_summary,
)
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain.callbacks import get_openai_callback
from process_runner import run_in_subprocess
import model_worker
from llm_models import get_model
from llm_adapters import create_condensation_adapter
from utils import remove_thinking_tokens, create_backup_file, parse_backup_file, list_backup_files
from email_sender import send_email_with_audio, send_email_with_attachments
from telegram_sender import send_telegram_with_audio, send_telegram_with_attachments

print(f"[INFO]    [{datetime.now().strftime('%H:%M:%S')}] ASR backend: {ASR_BACKEND} | TTS backend: {TTS_BACKEND}")

app = Flask(__name__)

# Global conversation state (in production, use session-based storage)
window_memory_100 = ConversationBufferWindowMemory(k=100)
conversation_chain = None
current_mode = None
session_history = InMemoryChatMessageHistory()
current_model_key = os.getenv("DEFAULT_MODEL_KEY", "mlx_community_qwen_stream_local_llm")
current_model = get_model(current_model_key)

def check_llm_server():
    """Check if the local LLM server is running"""
    _base = os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1").rstrip("/")
    try:
        response = requests.get(f"{_base}/models", timeout=2)
        return response.status_code == 200
    except:
        return False

def create_runnable_chain():
    """Create a runnable chain with mode-specific system prompt"""

    print(f"[INFO] [{datetime.now().strftime('%H:%M:%S')}] Creating runnable chain for YouTube mode")
    system_message = subject_matter_expert_prompt

    chat_prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_message),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])

    base_chain = chat_prompt_template | current_model | StrOutputParser()

    return RunnableWithMessageHistory(
        runnable=base_chain,  # 'runnable' is the required keyword
        get_session_history=lambda session_id: session_history,  # Lambda needs an arg
        input_messages_key="input",
        history_messages_key="history"
    )

@app.route('/')
def index():
    return render_template('index.html')

def _fetch_youtube_title(video_id: str) -> str:
    """Fetch YouTube video title via oEmbed. Returns empty string on failure."""
    try:
        import urllib.request
        import json as _json
        oembed_url = f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json"
        with urllib.request.urlopen(oembed_url, timeout=5) as resp:
            data = _json.loads(resp.read().decode())
            return data.get("title", "")
    except Exception:
        return ""

@app.route('/load_content', methods=['POST'])
def load_content():
    """Load content, condense it using condenser_service, and prepare for Q&A"""
    global conversation_chain, current_mode, current_model

    data = request.json
    url = data.get('url')

    mode = data.get('mode', 'youtube')
    auto_send_telegram = data.get('auto_send_telegram', False)  # Only true for auto-processor
    category = data.get('category', 'tech')  # 'tech', 'social', or 'science'
    fetch_mode = data.get('fetch_mode', 'transcript')  # 'transcript' or 'audio'

    if not url:
        return jsonify({'error': 'URL is required'}), 400

    if mode != 'youtube':
        return jsonify({'error': 'Invalid mode. Only "youtube" is supported.', 'success': False}), 400

    if fetch_mode not in ['transcript', 'audio']:
        print(f"[WARNING] Invalid fetch_mode '{fetch_mode}', defaulting to 'transcript'")
        fetch_mode = 'transcript'

    script_style = data.get('script_style', 'summary')
    if script_style not in ['summary', 'analysis']:
        print(f"[WARNING] Invalid script_style '{script_style}', defaulting to 'summary'")
        script_style = 'summary'

    if category not in ['tech', 'social', 'science']:
        print(f"[WARNING] Invalid category '{category}', defaulting to 'tech'")
        category = 'tech'

    # ------------------------------------------------------------------
    # Normalise YouTube URLs — strip playlist params, tracking params,
    # and everything else, reducing to bare https://www.youtube.com/watch?v=ID.
    # This guarantees a single canonical form is used for the cache key,
    # the stored checkpoint, yt-dlp, and the transcript API.
    # ------------------------------------------------------------------

    _vid = extract_video_id(url)
    if not _vid:
        return jsonify({'error': 'Invalid YouTube URL format.', 'success': False}), 400
    url = f"https://www.youtube.com/watch?v={_vid}"
    print(f"[INFO] [{datetime.now().strftime('%H:%M:%S')}] Normalised YouTube URL → {url}")

    # ------------------------------------------------------------------
    # Checkpoint setup — compute key before any I/O so every stage can
    # save progress and a retry resumes from where it stopped.
    # ------------------------------------------------------------------
    condensation_adapter = create_condensation_adapter(current_model, current_model_key)
    condensation_profile_id = get_condensation_profile_id(
        condensation_adapter, script_style=script_style
    )
    checkpoint_model_key = f"{current_model_key}|{condensation_profile_id}"
    checkpoint_key = compute_cache_key(
        url, mode, checkpoint_model_key, fetch_mode, script_style
    )
    checkpoint = load_checkpoint(checkpoint_key)
    if checkpoint is None:
        checkpoint_key, checkpoint = create_checkpoint(
            url, mode, checkpoint_model_key, fetch_mode, script_style
        )
    else:
        print(
            f"[INFO] [{datetime.now().strftime('%H:%M:%S')}] "
            f"Resuming from checkpoint {checkpoint_key} — "
            f"{get_progress_summary(checkpoint)}"
        )

    audio_time = None  # declared early so except blocks can reference it
    try:
        tts_input = ''  # declared early; set in both cache-hit and non-cache branches
        # Initialize conversation chain for the selected mode
        conversation_chain = create_runnable_chain()
        current_mode = mode

        # --------------------------------------------------------------
        # Full cache hit: audio already generated and file still on disk
        # --------------------------------------------------------------
        cached_audio_path = checkpoint.get("audio_file_path")
        _cache_hit = False
        # url is already normalised, so extract_video_id is stable here

        _video_id = extract_video_id(url)
        checkpoint["video_id"] = _video_id or ""
        if (
            cached_audio_path
            and Path(cached_audio_path).exists()
            and checkpoint.get("final_output")
        ):
            condensed_content = checkpoint["final_output"]
            tts_input = checkpoint.get("tts_input") or condensed_content
            audio_file = os.path.basename(cached_audio_path)
            audio_file_path = cached_audio_path
            raw_word_count = len((checkpoint.get("raw_content") or "").split())
            condensed_word_count = len(condensed_content.split())
            _cache_hit = True
            print(
                f"[INFO] [{datetime.now().strftime('%H:%M:%S')}] "
                f"Full cache hit — video_id={_video_id} audio={audio_file}"
            )
            # Skip straight to Telegram / response building below
        else:
            # -----------------------------------------------------------
            # Step 1: Fetch raw content (resume if already cached)
            # -----------------------------------------------------------
            if checkpoint.get("raw_content"):
                raw_content = checkpoint["raw_content"]
                print(
                    f"[INFO] [{datetime.now().strftime('%H:%M:%S')}] "
                    f"Resuming: raw_content already cached ({len(raw_content)} chars), "
                    f"source={checkpoint.get('source')}"
                )
            else:
                # url is already normalised to watch?v=ID — extract_video_id always succeeds here
                video_id = extract_video_id(url)

                if fetch_mode == 'audio':
                    # Audio queue: bypass transcript API, go straight to ASR backend.
                    # download_audio() reuses a cached mp3 if it already exists on disk,
                    # so a crash mid-transcription does not force a full re-download.
                    print(
                        f"[INFO] [{datetime.now().strftime('%H:%M:%S')}] "
                        f"Audio queue: transcribing via {ASR_BACKEND} for video: {video_id}"
                    )
                    raw_content = run_in_subprocess(model_worker.asr_worker, url, video_id)
                    if raw_content.startswith("Error:"):
                        print(f"[ERROR] ASR transcription failed: {raw_content}")
                        return jsonify({'error': raw_content, 'success': False}), 400
                    checkpoint["raw_content"] = raw_content
                    checkpoint["source"] = ASR_BACKEND
                    save_checkpoint(checkpoint_key, checkpoint)
                    print(f"[SUCCESS] {ASR_BACKEND} transcript obtained: {len(raw_content)} chars")
                else:
                    # Transcript queue: try transcript API (manual transcripts only),
                    # fall back to Whisper automatically inside get_youtube_transcript().
                    print(f"[INFO] [{datetime.now().strftime('%H:%M:%S')}] Fetching YouTube transcript from: {url}")
                    raw_content = get_youtube_transcript(url)
                    if (
                        raw_content.startswith("No transcript available")
                        or raw_content.startswith("Error fetching")
                        or raw_content.startswith("Invalid YouTube")
                        or raw_content.startswith("Error:")
                    ):
                        print(f"[ERROR] YouTube transcript fetch failed: {raw_content}")
                        return jsonify({'error': raw_content, 'success': False}), 400
                    checkpoint["raw_content"] = raw_content
                    checkpoint["source"] = "youtube_fetcher"
                    save_checkpoint(checkpoint_key, checkpoint)
                    print(f"[SUCCESS] YouTube transcript fetched: {len(raw_content)} chars")

            raw_word_count = len(raw_content.split())

            # -----------------------------------------------------------
            # Step 2: Condense (resume if final_output already cached)
            # -----------------------------------------------------------
            if checkpoint.get("final_output"):
                condensed_content = checkpoint["final_output"]
                print(
                    f"[INFO] [{datetime.now().strftime('%H:%M:%S')}] "
                    f"Resuming: final_output already cached ({len(condensed_content)} chars), "
                    f"skipping all condensation"
                )
            else:
                print(f"[INFO] [{datetime.now().strftime('%H:%M:%S')}] Condensing content...")
                condensed_content = condense_content(
                    raw_content, condensation_adapter, checkpoint_key, checkpoint,
                    script_style=script_style
                )
                print(
                    f"[SUCCESS] Condensed: {len(raw_content)} -> {len(condensed_content)} chars"
                )

            condensed_word_count = len(condensed_content.split())

            # -----------------------------------------------------------
            # Step 2b: Generate spoken intro from MAP summaries
            # -----------------------------------------------------------
            intro_text = ""
            try:
                if checkpoint.get("intro_text"):
                    intro_text = checkpoint["intro_text"]
                    print(f"[INFO]    [{datetime.now().strftime('%H:%M:%S')}] Intro loaded from cache")
                else:
                    map_results: dict = checkpoint.get("map_results", {})
                    valid_keys = [k for k in map_results if map_results[k]]
                    map_samples = []
                    if valid_keys:
                        sorted_keys = sorted(valid_keys, key=int)
                        sample_keys = sorted(random.sample(sorted_keys, min(3, len(sorted_keys))), key=int)
                        map_samples = [map_results[k] for k in sample_keys]
                        print(f"[INFO]    [{datetime.now().strftime('%H:%M:%S')}] Generating intro ({len(sample_keys)} MAP samples)...")
                    else:
                        print(f"[INFO]    [{datetime.now().strftime('%H:%M:%S')}] Generating intro from final script excerpts...")

                    intro_prompt = build_intro_prompt(
                        condensation_adapter,
                        map_samples,
                        condensed_content,
                        script_style=script_style,
                    )
                    if intro_prompt:
                        intro_text, tags_found = remove_thinking_tokens(
                            condensation_adapter.generate(intro_prompt)
                        )
                        if not tags_found:
                            print(f"[WARNING] Intro tags missing — continuing without intro")
                            intro_text = ""
                        else:
                            checkpoint["intro_text"] = intro_text
                            save_checkpoint(checkpoint_key, checkpoint)
                    else:
                        print(f"[WARNING] Intro prompt did not fit the configured context — continuing without intro")
            except Exception as e:
                print(f"[WARNING] Intro generation failed: {e} — continuing without intro")
                intro_text = ""

            tts_input = (intro_text + "\n\n" + condensed_content).strip() if intro_text else condensed_content
            checkpoint["tts_input"] = tts_input
            save_checkpoint(checkpoint_key, checkpoint)

            # -----------------------------------------------------------
            # Step 3: Generate audio (resume if already cached)
            # -----------------------------------------------------------
            audio_file = None
            audio_file_path = None
            print(
                f"[INFO] [{datetime.now().strftime('%H:%M:%S')}] "
                f"Generating audio ({len(condensed_content)} chars)..."
            )
            audio_start_time = time.time()

            # Select TTS voice once per video and persist to checkpoint for consistent resume
            tts_voice = checkpoint.get("tts_voice", "") if checkpoint else ""
            if not tts_voice and TTS_BACKEND == "voxtral":
                tts_voice = random.choice(["neutral_male"])
                if checkpoint:
                    checkpoint["tts_voice"] = tts_voice
                    save_checkpoint(checkpoint_key, checkpoint)
                print(f"[INFO]    [{datetime.now().strftime('%H:%M:%S')}] TTS voice selected: {tts_voice}")

            try:

                _yt_title = checkpoint.get("video_title", "") if checkpoint else ""
                if not _yt_title and _video_id:
                    _yt_title = _fetch_youtube_title(_video_id)
                    if _yt_title and checkpoint:
                        checkpoint["video_title"] = _yt_title
                        save_checkpoint(checkpoint_key, checkpoint)
                audio_file_path = run_in_subprocess(model_worker.tts_worker, tts_input, "", tts_voice, _video_id or "", _yt_title)
                audio_file = os.path.basename(audio_file_path)
                # Save audio path BEFORE building the response to avoid losing it on crash
                checkpoint["audio_file_path"] = str(audio_file_path)
                save_checkpoint(checkpoint_key, checkpoint)
                audio_time = time.time() - audio_start_time
                print(f"[SUCCESS] Audio generated: {audio_file}")
                print(f"[TIME] Audio generation took: {audio_time:.2f}s")
            except Exception as e:
                error_msg = f"Audio generation failed: {e}"
                print(f"[ERROR] {error_msg}")
                if auto_send_telegram:
                    return jsonify({'error': error_msg, 'success': False}), 422
                # For manual load, audio failure is not critical
                print(f"[WARNING] Continuing without audio")
        
        # Step 4: Send to Telegram only if auto_send_telegram is True
        if auto_send_telegram:
            print(f"[INFO] [{datetime.now().strftime('%H:%M:%S')}] Auto-sending to Telegram (category: {category})...")
            try:
                bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
                
                # Map category to channel ID
                channel_map = {
                    'tech': os.getenv('TELEGRAM_CHANNEL_TECH'),
                    'social': os.getenv('TELEGRAM_CHANNEL_SOCIAL'),
                    'science': os.getenv('TELEGRAM_CHANNEL_SCIENCE')
                }
                
                # Map category to discussion group (chat_id)
                chat_map = {
                    'tech': os.getenv('TELEGRAM_CHAT_ID_TECH') or os.getenv('TELEGRAM_CHAT_ID'),  # Backward compatibility
                    'social': os.getenv('TELEGRAM_CHAT_ID_SOCIAL'),
                    'science': os.getenv('TELEGRAM_CHAT_ID_SCIENCE')
                }
                
                # Get channel for this category, fallback to old TELEGRAM_CHANNEL_ID if specific not set
                channel_id = channel_map.get(category)
                if not channel_id:
                    channel_id = os.getenv('TELEGRAM_CHANNEL_ID')  # Backward compatibility
                    print(f"[WARNING] TELEGRAM_CHANNEL_{category.upper()} not set, using TELEGRAM_CHANNEL_ID")
                
                # Get discussion group for this category
                chat_id = chat_map.get(category)
                if not chat_id:
                    print(f"[ERROR] TELEGRAM_CHAT_ID_{category.upper()} not set")
                    return jsonify({'error': f'Discussion group for {category} not configured', 'success': False}), 422
                
                if not chat_id or not bot_token:
                    error_msg = "Telegram credentials not configured (TELEGRAM_CHAT_ID and TELEGRAM_BOT_TOKEN required)"
                    print(f"[ERROR] {error_msg}")
                    return jsonify({'error': error_msg, 'success': False}), 422
                
                if not audio_file_path:
                    error_msg = "Cannot send to Telegram: Audio file not generated"
                    print(f"[ERROR] {error_msg}")
                    return jsonify({'error': error_msg, 'success': False}), 422
                
                content_type = 'YouTube Video'
                message = f"📝 Condensed {content_type}\n\n{condensed_content}"
                
                print(f"[TELEGRAM] Sending to channel: {channel_id}, discussion group: {chat_id}")
                telegram_success = send_telegram_with_audio(
                    chat_id=chat_id,
                    message=message,
                    audio_file_path=audio_file_path,
                    bot_token=bot_token,
                    source_url=url,
                    channel_id=channel_id if channel_id else None
                )
                
                if not telegram_success:
                    error_msg = "Failed to send content to Telegram"
                    print(f"[ERROR] {error_msg}")
                    # Create backup file
                    try:
                        backup_path = create_backup_file(url, condensed_content, audio_file_path, category)
                        error_msg = f"Failed to send to Telegram. Backup created at: {backup_path}"
                        print(f"[BACKUP] {error_msg}")
                    except Exception as backup_error:
                        print(f"[ERROR] Failed to create backup: {backup_error}")
                    return jsonify({'error': error_msg, 'success': False}), 422
                
                print(f"[SUCCESS] Content sent to Telegram successfully")
            except Exception as e:
                error_msg = f"Telegram sending failed: {e}"
                print(f"[ERROR] {error_msg}")
                # Create backup file
                try:
                    backup_path = create_backup_file(url, condensed_content, audio_file_path, category)
                    error_msg = f"Telegram error: {e}. Backup created at: {backup_path}"
                    print(f"[BACKUP] {error_msg}")
                except Exception as backup_error:
                    print(f"[ERROR] Failed to create backup: {backup_error}")
                return jsonify({'error': error_msg, 'success': False}), 422
        
        # Step 5: Store condensed content in conversation memory for future Q&A
        # The memory now contains: system prompt (from create_runnable_chain) + condensed input
        session_history.add_user_message(f"Here is the condensed video transcript content (Original: {raw_word_count} words, Condensed: {condensed_word_count} words):\n\n{condensed_content}")
        session_history.add_ai_message(f"I have received and processed the condensed content. I'm ready to answer your questions about it.")
        print(f"[INFO] [{datetime.now().strftime('%H:%M:%S')}] Condensed content added to memory. Ready for Q&A.")

        # Return success only if everything succeeded
        return jsonify({
            'content': condensed_content,
            'mode': mode,
            'word_count': condensed_word_count,
            'original_word_count': raw_word_count,
            'audio_file': audio_file,
            'audio_time': round(audio_time, 2) if audio_time else None,
            'from_cache': _cache_hit,
            'video_id': _video_id,
            'tts_script': tts_input,

            'checkpoint_key': checkpoint_key,
            'success': True
        })

    except ValueError as e:
        # ValueError is raised when thinking tokens aren't removed properly.
        # The checkpoint has already been saved with all progress so far;
        # the next retry will resume from the last successful step.
        print(f"[ERROR] Thinking token validation failed: {e}")
        return jsonify({
            'error': str(e),
            'success': False,
            'checkpoint_key': checkpoint_key,
            'resume_progress': get_progress_summary(checkpoint),
        }), 422
    except Exception as e:
        print(f"[ERROR] load_content failed: {e}")
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/send_email', methods=['POST'])
def send_email():
    """Send condensed content with audio via email"""
    data = request.json

    audio_file = data.get('audio_file')
    content = data.get('content')
    url = data.get('url', '')
    
    # Get recipient email from environment variable
    recipient_email = os.getenv('RECIPIENT_GMAIL_ADDRESS')
    
    if not recipient_email:
        print("[ERROR] RECIPIENT_GMAIL_ADDRESS environment variable not set")
        return jsonify({'error': 'Recipient email not configured. Set RECIPIENT_GMAIL_ADDRESS environment variable.', 'success': False}), 500
    
    if not audio_file:
        return jsonify({'error': 'Audio file not found', 'success': False}), 400
    
    try:
        # Build audio file path
        audio_file_path = os.path.join('kokoro_outputs', audio_file)
        
        if not os.path.exists(audio_file_path):
            return jsonify({'error': 'Audio file does not exist', 'success': False}), 404
        
        # Create email subject and body
        content_type = 'Video Transcript'
        subject = f'Condensed {content_type} with Audio'
        
        # Truncate content if too long for email body
        body_content = content[:2000] + '...' if len(content) > 2000 else content
        
        # Include source URL in email body
        url_section = f"\n\nSource URL: {url}\n" if url else ""
        body_text = f"Here is the condensed {content_type.lower()}:{url_section}\n\n{body_content}\n\nPlease find the audio file attached."
        
        print(f"[INFO] [{datetime.now().strftime('%H:%M:%S')}] Sending email to {recipient_email}...")
        success = send_email_with_audio(
            recipient_email=recipient_email,
            subject=subject,
            body_text=body_text,
            audio_file_path=audio_file_path
        )
        
        if success:
            print(f"[SUCCESS] Email sent to {recipient_email}")
            return jsonify({'success': True, 'message': 'Email sent successfully'})
        else:
            print(f"[ERROR] Failed to send email to {recipient_email}")
            return jsonify({'error': 'Failed to send email. Check server logs for details.', 'success': False}), 500
            
    except Exception as e:
        print(f"[ERROR] send_email endpoint failed: {e}")
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/send_telegram', methods=['POST'])
def send_telegram():
    """Send condensed content with audio via Telegram"""
    data = request.json

    audio_file = data.get('audio_file')
    content = data.get('content')
    url = data.get('url', '')
    
    # Get Telegram credentials from environment variables
    bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
    chat_id = os.getenv('TELEGRAM_CHAT_ID')
    channel_id = os.getenv('TELEGRAM_CHANNEL_ID')  # Optional channel for posting
    
    if not bot_token:
        print("[ERROR] TELEGRAM_BOT_TOKEN environment variable not set")
        return jsonify({'error': 'Telegram bot token not configured. Set TELEGRAM_BOT_TOKEN environment variable.', 'success': False}), 500
    
    if not chat_id:
        print("[ERROR] TELEGRAM_CHAT_ID environment variable not set")
        return jsonify({'error': 'Telegram chat ID not configured. Set TELEGRAM_CHAT_ID environment variable.', 'success': False}), 500
    
    if not audio_file:
        return jsonify({'error': 'Audio file not found', 'success': False}), 400
    
    try:
        # Build audio file path
        audio_file_path = os.path.join('kokoro_outputs', audio_file)
        
        if not os.path.exists(audio_file_path):
            return jsonify({'error': 'Audio file does not exist', 'success': False}), 404
        
        # Create message text
        content_type = 'Video Transcript'
        
        # Don't include URL in message since it will be sent separately
        message = f"📝 Condensed {content_type}\n\n{content}"
        
        print(f"[INFO] [{datetime.now().strftime('%H:%M:%S')}] Sending to Telegram chat {chat_id}...")
        print(f"[DEBUG] Message length: {len(message)} characters")
        success = send_telegram_with_audio(
            chat_id=chat_id,
            message=message,
            audio_file_path=audio_file_path,
            bot_token=bot_token,
            source_url=url,
            channel_id=channel_id if channel_id else None
        )
        
        if success:
            print(f"[SUCCESS] Telegram message sent to {chat_id}")
            return jsonify({'success': True, 'message': 'Telegram message sent successfully'})
        else:
            print(f"[ERROR] Failed to send Telegram message to {chat_id}")
            return jsonify({'error': 'Failed to send Telegram message. Check server logs for details.', 'success': False}), 500
            
    except Exception as e:
        print(f"[ERROR] send_telegram endpoint failed: {e}")
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/retry_failed_telegrams', methods=['POST'])
def retry_failed_telegrams():
    """Retry sending all failed Telegram messages from backup_content folder"""
    print("[RETRY] Starting retry of failed Telegram messages...")
    
    # Get Telegram credentials
    bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
    
    if not bot_token:
        return jsonify({
            'error': 'Telegram not configured. Set TELEGRAM_BOT_TOKEN environment variable.',
            'success': False
        }), 500
    
    # Create channel map for category-specific channels
    channel_map = {
        'tech': os.getenv('TELEGRAM_CHANNEL_TECH'),
        'social': os.getenv('TELEGRAM_CHANNEL_SOCIAL'),
        'science': os.getenv('TELEGRAM_CHANNEL_SCIENCE')
    }
    
    # Create chat map for category-specific discussion groups
    chat_map = {
        'tech': os.getenv('TELEGRAM_CHAT_ID_TECH') or os.getenv('TELEGRAM_CHAT_ID'),  # Backward compatibility
        'social': os.getenv('TELEGRAM_CHAT_ID_SOCIAL'),
        'science': os.getenv('TELEGRAM_CHAT_ID_SCIENCE')
    }
    
    try:
        # Create sent subfolder if it doesn't exist
        sent_dir = Path('backup_content') / 'sent'
        sent_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all backup files
        backup_files = list_backup_files()
        
        if not backup_files:
            return jsonify({
                'success': True,
                'total': 0,
                'sent': 0,
                'skipped': 0,
                'failed': 0,
                'message': 'No backup files found'
            })
        
        sent_count = 0
        skipped_count = 0
        failed_count = 0
        skipped_files = []
        failed_files = []
        
        # Process each backup file
        for backup_file in backup_files:
            try:
                print(f"[RETRY] Processing {backup_file.name}...")
                
                # Parse backup file
                backup_data = parse_backup_file(str(backup_file))
                if not backup_data:
                    print(f"[RETRY] Failed to parse {backup_file.name}, skipping")
                    skipped_count += 1
                    skipped_files.append(backup_file.name)
                    continue
                
                # Check if audio file exists
                audio_file_path = backup_data['audio_file_path']
                if not os.path.exists(audio_file_path):
                    print(f"[RETRY] Audio file not found: {audio_file_path}, skipping")
                    skipped_count += 1
                    skipped_files.append(f"{backup_file.name} (audio missing)")
                    continue
                
                # Get category from backup data (defaults to 'tech' for old backups)
                category = backup_data.get('category', 'tech')
                
                # Get channel for this category
                channel_id = channel_map.get(category)
                if not channel_id:
                    channel_id = os.getenv('TELEGRAM_CHANNEL_ID')  # Backward compatibility
                    print(f"[RETRY] TELEGRAM_CHANNEL_{category.upper()} not set, using TELEGRAM_CHANNEL_ID")
                
                # Get discussion group for this category
                chat_id = chat_map.get(category)
                if not chat_id:
                    print(f"[RETRY] TELEGRAM_CHAT_ID_{category.upper()} not set, skipping {backup_file.name}")
                    skipped_count += 1
                    skipped_files.append(f"{backup_file.name} (chat_id missing)")
                    continue
                
                # Send to Telegram
                url = backup_data['url']
                content = backup_data['content']
                message = f"📝 Condensed Content\n\n{content}"
                
                print(f"[RETRY] Sending to channel: {channel_id}, discussion group: {chat_id}, category: {category}")
                success = send_telegram_with_audio(
                    chat_id=chat_id,
                    message=message,
                    audio_file_path=audio_file_path,
                    bot_token=bot_token,
                    source_url=url,
                    channel_id=channel_id if channel_id else None
                )
                
                if success:
                    print(f"[RETRY] Successfully sent {backup_file.name}")
                    sent_count += 1
                    
                    # Move backup file to sent subfolder
                    dest_path = sent_dir / backup_file.name
                    shutil.move(str(backup_file), str(dest_path))
                    print(f"[RETRY] Moved {backup_file.name} to sent/")
                else:
                    print(f"[RETRY] Failed to send {backup_file.name}")
                    failed_count += 1
                    failed_files.append(backup_file.name)
                    
            except Exception as e:
                print(f"[RETRY] Error processing {backup_file.name}: {e}")
                failed_count += 1
                failed_files.append(f"{backup_file.name} ({str(e)})")
        
        # Return summary
        result = {
            'success': True,
            'total': len(backup_files),
            'sent': sent_count,
            'skipped': skipped_count,
            'failed': failed_count,
            'skipped_files': skipped_files,
            'failed_files': failed_files
        }
        
        print(f"[RETRY] Complete. Sent: {sent_count}, Skipped: {skipped_count}, Failed: {failed_count}")
        return jsonify(result)
        
    except Exception as e:
        print(f"[ERROR] retry_failed_telegrams endpoint failed: {e}")
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/send_quick_email', methods=['POST'])
def send_quick_email():
    """Send quick email with custom message and attachments from UI"""
    message = request.form.get('message')
    files = request.files.getlist('attachments')
    
    # Get recipient email from environment variable
    recipient_email = os.getenv('RECIPIENT_GMAIL_ADDRESS')
    
    if not recipient_email:
        print("[ERROR] RECIPIENT_GMAIL_ADDRESS environment variable not set")
        return jsonify({'error': 'Recipient email not configured. Set RECIPIENT_GMAIL_ADDRESS environment variable.', 'success': False}), 500
    
    if not message:
        return jsonify({'error': 'Message is required', 'success': False}), 400
    
    try:
        # Save uploaded files temporarily
        temp_dir = 'temp_attachments'
        os.makedirs(temp_dir, exist_ok=True)
        
        attachment_paths = []
        for file in files:
            if file.filename:
                file_path = os.path.join(temp_dir, file.filename)
                file.save(file_path)
                attachment_paths.append(file_path)
                print(f"[INFO] [{datetime.now().strftime('%H:%M:%S')}] Saved temporary attachment: {file.filename}")
        
        # Send email
        subject = 'Quick Message from Content Analyzer'
        
        print(f"[INFO] [{datetime.now().strftime('%H:%M:%S')}] Sending quick email to {recipient_email}...")
        success = send_email_with_attachments(
            recipient_email=recipient_email,
            subject=subject,
            body_text=message,
            attachment_paths=attachment_paths if attachment_paths else None
        )
        
        # Clean up temporary files
        for file_path in attachment_paths:
            try:
                os.remove(file_path)
                print(f"[INFO] [{datetime.now().strftime('%H:%M:%S')}] Cleaned up temporary file: {file_path}")
            except Exception as e:
                print(f"[WARNING] Failed to remove temp file {file_path}: {e}")
        
        # Remove temp directory if empty
        try:
            os.rmdir(temp_dir)
        except:
            pass
        
        if success:
            print(f"[SUCCESS] Quick email sent to {recipient_email}")
            return jsonify({'success': True, 'message': 'Email sent successfully'})
        else:
            print(f"[ERROR] Failed to send quick email to {recipient_email}")
            return jsonify({'error': 'Failed to send email. Check server logs for details.', 'success': False}), 500
            
    except Exception as e:
        print(f"[ERROR] send_quick_email endpoint failed: {e}")
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/send_quick_telegram', methods=['POST'])
def send_quick_telegram():
    """Send quick Telegram message with custom text and attachments from UI"""
    message = request.form.get('message')
    files = request.files.getlist('attachments')
    
    # Get Telegram credentials from environment variables
    bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
    chat_id = os.getenv('TELEGRAM_CHAT_ID')
    
    if not bot_token:
        print("[ERROR] TELEGRAM_BOT_TOKEN environment variable not set")
        return jsonify({'error': 'Telegram bot token not configured. Set TELEGRAM_BOT_TOKEN environment variable.', 'success': False}), 500
    
    if not chat_id:
        print("[ERROR] TELEGRAM_CHAT_ID environment variable not set")
        return jsonify({'error': 'Telegram chat ID not configured. Set TELEGRAM_CHAT_ID environment variable.', 'success': False}), 500
    
    if not message:
        return jsonify({'error': 'Message is required', 'success': False}), 400
    
    try:
        # Save uploaded files temporarily
        temp_dir = 'temp_attachments'
        os.makedirs(temp_dir, exist_ok=True)
        
        attachment_paths = []
        for file in files:
            if file.filename:
                file_path = os.path.join(temp_dir, file.filename)
                file.save(file_path)
                attachment_paths.append(file_path)
                print(f"[INFO] [{datetime.now().strftime('%H:%M:%S')}] Saved temporary attachment: {file.filename}")
        
        # Send Telegram message
        print(f"[INFO] [{datetime.now().strftime('%H:%M:%S')}] Sending quick Telegram message to {chat_id}...")
        success = send_telegram_with_attachments(
            chat_id=chat_id,
            message=message,
            attachment_paths=attachment_paths if attachment_paths else None,
            bot_token=bot_token
        )
        
        # Clean up temporary files
        for file_path in attachment_paths:
            try:
                os.remove(file_path)
                print(f"[INFO] [{datetime.now().strftime('%H:%M:%S')}] Cleaned up temporary file: {file_path}")
            except Exception as e:
                print(f"[WARNING] Failed to remove temp file {file_path}: {e}")
        
        # Remove temp directory if empty
        try:
            os.rmdir(temp_dir)
        except:
            pass
        
        if success:
            print(f"[SUCCESS] Quick Telegram message sent to {chat_id}")
            return jsonify({'success': True, 'message': 'Telegram message sent successfully'})
        else:
            print(f"[ERROR] Failed to send quick Telegram message to {chat_id}")
            return jsonify({'error': 'Failed to send Telegram message. Check server logs for details.', 'success': False}), 500
            
    except Exception as e:
        print(f"[ERROR] send_quick_telegram endpoint failed: {e}")
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/chat', methods=['POST'])
def chat():
    """Handle follow-up questions"""
    data = request.json
    user_input = data.get('message')
    generate_audio_flag = data.get('generate_audio', False)

    if not user_input:
        return jsonify({'error': 'Message is required'}), 400

    try:
        # Get response from conversation chain
        print(f"[INFO] [{datetime.now().strftime('%H:%M:%S')}] Generating LLM response for: '{user_input[:50]}...'")
        
        # Track token usage with callback and time
        import time
        llm_start_time = time.time()
        
        with get_openai_callback() as cb:
            response = conversation_chain.invoke({
                "input": user_input
            })
            
            # Extract token usage from callback
            token_usage = {
                'prompt_tokens': cb.prompt_tokens,
                'completion_tokens': cb.completion_tokens,
                'total_tokens': cb.total_tokens
            }
        
        llm_time = time.time() - llm_start_time
        response_text = response['response']
        
        # Remove thinking tokens from response
        response_text, thinking_tokens_removed = remove_thinking_tokens(response_text)
        if not thinking_tokens_removed:
            error_msg = "Failed to remove thinking tokens from LLM response"
            print(f"[ERROR] {error_msg}")
            return jsonify({'error': error_msg, 'success': False}), 422
        
        print(f"[SUCCESS] LLM response generated. Length: {len(response_text)} characters")
        print(f"[TOKENS] Prompt: {token_usage['prompt_tokens']}, Completion: {token_usage['completion_tokens']}, Total: {token_usage['total_tokens']}")
        print(f"[TIME] LLM generation took: {llm_time:.2f}s")
        
        # Get token usage if available
        token_usage = None
        # if hasattr(groq_llm, 'last_response') and groq_llm.last_response:
        #     usage = groq_llm.last_response.get('usage', {})
        #     if usage:
        #         token_usage = {
        #             'prompt_tokens': usage.get('prompt_tokens', 0),
        #             'completion_tokens': usage.get('completion_tokens', 0),
        #             'total_tokens': usage.get('total_tokens', 0)
        #         }
        
        audio_file = None
        audio_time = None

        # Generate audio if requested
        if generate_audio_flag:
            try:
                print(f"[INFO] [{datetime.now().strftime('%H:%M:%S')}] Generating audio for response ({len(response_text)} chars)...")
                audio_start_time = time.time()
                
                audio_file_path = run_in_subprocess(model_worker.tts_worker, response_text, "")
                audio_file = os.path.basename(audio_file_path)
                
                audio_time = time.time() - audio_start_time
                print(f"[SUCCESS] Audio generated: {audio_file}")
                print(f"[TIME] Audio generation took: {audio_time:.2f}s")
            except Exception as e:
                print(f"[ERROR] Audio generation failed: {e}")

        return jsonify({
            'response': response_text,
            'audio_file': audio_file,
            'token_usage': token_usage,
            'llm_time': round(llm_time, 2),
            'audio_time': round(audio_time, 2) if audio_time else None,
            'success': True
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/streamChat', methods=['POST'])
def stream_chat():
    """Handle follow-up questions with streaming"""
    print(f"[INFO] [{datetime.now().strftime('%H:%M:%S')}] streamChat endpoint called")
    # Check if LLM server is accessible first
    if not check_llm_server():
        print("[ERROR] LLM server not accessible")
        return jsonify({
            'error': f'Local LLM server is not running. Please start LM Studio and load a model at {os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1")}'
        }), 503

    data = request.json
    user_input = data.get('message')
    generate_audio_flag = data.get('generate_audio', False)
    print(f"[INFO] [{datetime.now().strftime('%H:%M:%S')}] User input: '{user_input[:50]}...' | Audio: {generate_audio_flag}")

    if not user_input:
        return jsonify({'error': 'Message is required'}), 400

    def stream_response():
        try:
            print("[DEBUG] stream_response: Starting generator")
            # Track token usage with callback and time
            import time
            
            token_usage = None
            audio_file = None
            audio_time = None

            llm_start_time = time.time()
            total_chunk_size = 0
            complete_response_text = ""

            print(f"[INFO] [{datetime.now().strftime('%H:%M:%S')}] Generating streaming LLM response for: '{user_input[:50]}...'")

            with get_openai_callback() as cb:
                chunk_received = False
                # with get_openai_callback() as cb:
                for chunk_response in conversation_chain.stream({
                    "input": user_input
                }, config = {"configurable": {"session_id": "any_string_here"}}):

                    if chunk_response:  # Only send non-empty chunks
                        chunk_received = True
                        complete_response_text += chunk_response
                        total_chunk_size += len(chunk_response)
                        # Send only the chunk during streaming
                        chunk_data = {'chunk': chunk_response}
                        yield f"data:{json.dumps(chunk_data)}\n\n"

                if not chunk_received:
                    raise Exception("No response from LLM - check if local LLM server is running")

                # Extract token usage from callback
                token_usage = {
                    'prompt_tokens': cb.prompt_tokens,
                    'completion_tokens': cb.completion_tokens,
                    'total_tokens': cb.total_tokens
                }

            llm_time = time.time() - llm_start_time
            print(f"[INFO] [{datetime.now().strftime('%H:%M:%S')}] Streaming complete. Time: {llm_time:.2f}s, Chunks: {total_chunk_size} chars")

            # Remove thinking tokens from complete response
            complete_response_text, thinking_tokens_removed = remove_thinking_tokens(complete_response_text)
            if not thinking_tokens_removed:
                error_msg = "Failed to remove thinking tokens from LLM response"
                print(f"[ERROR] {error_msg}")
                error_data = {'error': error_msg}
                yield f"data:{json.dumps(error_data)}\n\n"
                return
            print(f"[DEBUG] After thinking token removal: {len(complete_response_text)} chars")

            # print(f"[SUCCESS] LLM streaming response completed. Length: {total_chunk_size} characters")
            # print(f"[TOKENS] Prompt: {token_usage['prompt_tokens']}, Completion: {token_usage['completion_tokens']}, Total: {token_usage['total_tokens']}")
            # print(f"[TIME] LLM generation took: {llm_time:.2f}s")

            # Send streaming-done event if audio generation is requested
            if generate_audio_flag:
                streaming_done_data = {'streaming_done': True}
                yield f"data:{json.dumps(streaming_done_data)}\n\n"

            # Generate audio if requested
            if generate_audio_flag:
                try:
                    print(f"[INFO] [{datetime.now().strftime('%H:%M:%S')}] Generating audio for response ({len(complete_response_text)} chars)...")
                    audio_start_time = time.time()

                    audio_file_path = run_in_subprocess(model_worker.tts_worker, complete_response_text, "")
                    audio_file = os.path.basename(audio_file_path)

                    audio_time = time.time() - audio_start_time
                    print(f"[SUCCESS] Audio generated: {audio_file}")
                    print(f"[TIME] Audio generation took: {audio_time:.2f}s")
                except Exception as e:
                    print(f"[ERROR] Audio generation failed: {e}")

            # Send final metadata message
            final_data = {
                'done': True,
                'audio_file': audio_file,
                'token_usage': token_usage,
                'llm_time': round(llm_time, 2),
                'audio_time': round(audio_time, 2) if audio_time else None,
                'success': True
            }

            yield f"data:{json.dumps(final_data)}\n\n"
            
        except Exception as e:
            print(f"[ERROR] Stream chat error: {e}")
            error_data = {'error': str(e), 'done': True, 'success': False}
            yield f"data:{json.dumps(error_data)}\n\n"
    
    return Response(stream_with_context(stream_response()),
                    mimetype='text/event-stream',
                    headers={'X-Accel-Buffering': 'no',
                         'Cache-Control': 'no-cache'
                    })

@app.route('/audio/<filename>', methods=['GET'])
def get_audio(filename):
    """Serve a specific audio file"""
    print(f"[INFO] [{datetime.now().strftime('%H:%M:%S')}] Audio file requested: {filename}")
    audio_dir = 'kokoro_outputs'
    file_path = os.path.join(audio_dir, filename)

    if os.path.exists(file_path):
        print(f"[DEBUG] Serving audio file: {file_path}")
        return send_file(file_path, mimetype='audio/wav')

    print(f"[ERROR] Audio file not found: {file_path}")
    return jsonify({'error': 'Audio file not found'}), 404

@app.route('/clear_conversation', methods=['POST'])
def clear_conversation():
    """Clear conversation memory and reset mode"""
    print(f"[INFO] [{datetime.now().strftime('%H:%M:%S')}] Clearing conversation memory")
    global conversation_chain, current_mode
    
    if session_history:
        session_history.clear()
        print("[DEBUG] Session history cleared")
    
    conversation_chain = None
    current_mode = None
    print(f"[INFO] [{datetime.now().strftime('%H:%M:%S')}] Conversation state reset")

    return jsonify({'success': True, 'message': 'Conversation cleared'})

@app.route('/text_to_audio', methods=['POST'])
def text_to_audio():
    """Convert text to audio file"""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({
                'success': False,
                'error': 'Text is required'
            }), 400
        
        print(f"[INFO] [{datetime.now().strftime('%H:%M:%S')}] text_to_audio: Converting {len(text)} chars to audio")
        print(f"[DEBUG] Text preview: {text[:100]}...")

        audio_file_path = run_in_subprocess(model_worker.tts_worker, text, "")
        audio_filename = os.path.basename(audio_file_path)
        
        print(f"[SUCCESS] Audio file created: {audio_filename}")
        
        return jsonify({
            'success': True,
            'audio_file': audio_filename,
            'message': f'Audio generated successfully ({len(text)} chars)'
        })
        
    except Exception as e:
        error_msg = f"Text-to-audio conversion failed: {str(e)}"
        print(f"[ERROR] {error_msg}")
        return jsonify({
            'success': False,
            'error': error_msg
        }), 500

@app.route('/produce_video', methods=['POST'])

def produce_video_route():

    try:
        data = request.get_json(force=True) or {}
        # Enforce basename only — prevents path traversal (e.g. ../../etc/passwd)
        audio_file = os.path.basename(data.get('audio_file', '').strip())
        content = data.get('content', '').strip()
        title = data.get('title', '').strip()
        category = data.get('category', 'tech')
        url = data.get('url', '')

        if not audio_file or not content:
            return jsonify({'success': False, 'error': 'audio_file and content are required'}), 400

        if category not in ('tech', 'social', 'science'):
            category = 'tech'

        from video_producer import produce_video

        from youtube_uploader import upload_video, register_upload

        # Extract YouTube video_id so produce_video can download the original thumbnail
        video_id = extract_video_id(url) if url else ""
        result = produce_video(audio_file_basename=audio_file, script=content, title=title or 'Untitled', video_id=video_id or "")

        description = f"Condensed content from {url or 'unknown source'}\n\nCategory: {category}"
        yt_video_id = upload_video(
            mp4_path=result['mp4_path'],
            title=title or 'Untitled',
            description=description,
            thumb_path=result['thumb_path'],
            srt_path=result['srt_path'],
            category=category,
        )

        if not yt_video_id:
            return jsonify({'success': False, 'error': 'upload_video returned no video ID'}), 500

        try:
            register_upload({
                "yt_video_id": yt_video_id,
                "source_video_id": video_id or "",
                "url": url or "",
                "title": title or "Untitled",
                "category": category,
                "mp4_path": result['mp4_path'],
                "thumb_path": result['thumb_path'],
                "srt_path": result['srt_path'],
                "uploaded_at": datetime.now(timezone.utc).isoformat(),
                "published": False,
                "published_at": None,
            })
        except Exception as reg_err:
            print(f"[WARNING] register_upload failed (non-fatal): {reg_err}")

        print(f"[INFO]    [{datetime.now().strftime('%H:%M:%S')}] Video produced and uploaded: {yt_video_id}")
        return jsonify({
            'success': True,
            'yt_video_id': yt_video_id,
            'mp4_path': result['mp4_path'],
            'duration_seconds': result['duration_seconds'],
        })
    except Exception as e:
        print(f"[ERROR]   /produce_video failed: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/publish_youtube', methods=['POST'])

def publish_youtube_route():

    try:
        data = request.get_json(force=True) or {}
        yt_video_id = data.get('yt_video_id', '').strip()

        if not yt_video_id:
            return jsonify({'success': False, 'error': 'yt_video_id is required'}), 400

        from youtube_uploader import check_and_publish, mark_published
        result = check_and_publish(yt_video_id)

        if result.get("published"):
            try:
                mark_published(yt_video_id)
            except Exception as mp_err:
                print(f"[WARNING] mark_published failed (non-fatal): {mp_err}")

        print(f"[INFO]    [{datetime.now().strftime('%H:%M:%S')}] Manual publish result for {yt_video_id}: {result}")
        return jsonify({'success': True, 'result': result})
    except Exception as e:
        print(f"[ERROR]   /publish_youtube failed: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/youtube_uploads', methods=['GET'])
def youtube_uploads_route():
    try:
        from youtube_uploader import list_uploads
        uploads = list_uploads()
        return jsonify({'success': True, 'uploads': uploads})
    except Exception as e:
        print(f"[ERROR]   /youtube_uploads failed: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/clear_audio', methods=['POST'])
def clear_audio():
    data = request.json or {}
    checkpoint_key = data.get('checkpoint_key', '').strip()
    if not checkpoint_key:
        return jsonify({'success': False, 'error': 'checkpoint_key required'}), 400
    checkpoint = load_checkpoint(checkpoint_key)
    if checkpoint is None:
        return jsonify({'success': False, 'error': 'Checkpoint not found'}), 404
    checkpoint['audio_file_path'] = None
    save_checkpoint(checkpoint_key, checkpoint)
    print(f"[INFO]    [{datetime.now().strftime('%H:%M:%S')}] Cleared audio_file_path for checkpoint {checkpoint_key}")
    return jsonify({'success': True})

if __name__ == '__main__':
    purge_expired_translation_cache()
    app.run(host="0.0.0.0", port=5000)