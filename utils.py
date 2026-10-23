import re
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional


def format_elapsed(seconds: float) -> str:
    """Format a duration as seconds, or minutes plus leftover seconds."""
    value = max(0.0, float(seconds))
    total = int(round(value))
    if total < 60:
        if value < 10:
            return f"{value:.1f}s"
        return f"{total}s"
    minutes, secs = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    return f"{minutes}m {secs}s"


def remove_thinking_tokens(text: str) -> tuple[str, bool]:
    """
    Extract final script from LLM response by finding content within <final_script> tags.
    
    The LLM returns the final cleaned output wrapped in <final_script></final_script> tags.
    This function extracts only the content between the LAST opening and LAST closing tag,
    discarding all reasoning, thinking tokens, and other artifacts.

    Args:
        text: Raw LLM response text containing <final_script> tags

    Returns:
        Tuple of (cleaned_text, success_bool) where success_bool indicates if tags were found
    """
    if not text:
        print("[DEBUG] remove_thinking_tokens: Empty text received")
        return text, False

    print(f"[DEBUG] remove_thinking_tokens: Processing {len(text)} characters")
    original_length = len(text)

    # Find the last occurrence of opening and closing tags
    opening_tag = '<final_script>'
    closing_tag = '</final_script>'
    
    last_open_index = text.lower().rfind(opening_tag.lower())
    last_close_index = text.lower().rfind(closing_tag.lower())
    
    if last_open_index != -1 and last_close_index != -1 and last_close_index > last_open_index:
        # Extract content between last opening tag and last closing tag
        start_pos = last_open_index + len(opening_tag)
        final_content = text[start_pos:last_close_index].strip()
        
        print(f"[INFO] [{datetime.now().strftime('%H:%M:%S')}] Found <final_script> tags, extracting content from last occurrence")
        print(f"[CLEANUP] Extracted {len(final_content)} chars from final_script tag (removed {original_length - len(final_content)} chars)")
        return final_content, True
    else:
        print("[WARNING] No valid <final_script> tags found, thinking tokens not properly removed")
        print(f"[DEBUG] Original text content:\n{text}")
        return text.strip(), False


def create_backup_file(url: str, content: str, audio_file_path: str, category: str = 'tech') -> str:
    """
    Create a backup file for content and audio when Telegram sending fails.
    
    Args:
        url: The source URL
        content: The condensed content text
        audio_file_path: Path to the audio file
        category: Queue category (tech/social/science), defaults to 'tech'
        
    Returns:
        Path to the created backup file
    """
    # Create backup directory if it doesn't exist
    backup_dir = Path("backup_content")
    backup_dir.mkdir(exist_ok=True)
    
    # Generate a safe filename from the URL
    # Remove protocol and special characters
    safe_name = re.sub(r'https?://', '', url)
    safe_name = re.sub(r'[^\w\-_]', '_', safe_name)
    
    # Limit filename length
    if len(safe_name) > 100:
        safe_name = safe_name[:100]
    
    # Add timestamp to make it unique
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{safe_name}_{timestamp}.txt"
    
    backup_path = backup_dir / filename
    
    # Write content and audio path to the backup file
    with open(backup_path, 'w', encoding='utf-8') as f:
        f.write(f"Source URL: {url}\n")
        f.write(f"Category: {category}\n")
        f.write(f"Backup Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*80}\n\n")
        f.write(content)
        f.write(f"\n\n{'='*80}\n")
        f.write(f"Audio File Path: {audio_file_path}\n")
    
    print(f"[BACKUP] Created backup file: {backup_path} (category: {category})")
    return str(backup_path)


def parse_backup_file(file_path: str) -> Optional[dict]:
    """
    Parse a backup file to extract URL, content, audio file path, and category.
    
    Args:
        file_path: Path to the backup file
        
    Returns:
        Dictionary with 'url', 'content', 'audio_file_path', 'timestamp', 'category', or None if parsing fails
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract URL (first line format: "Source URL: <url>")
        url_match = re.search(r'^Source URL:\s*(.+)$', content, re.MULTILINE)
        if not url_match:
            print(f"[BACKUP PARSE] Failed to find URL in {file_path}")
            return None
        url = url_match.group(1).strip()
        
        # Extract category (optional, defaults to 'tech' for backward compatibility)
        category_match = re.search(r'^Category:\s*(.+)$', content, re.MULTILINE)
        category = category_match.group(1).strip() if category_match else 'tech'
        
        # Extract timestamp
        timestamp_match = re.search(r'^Backup Created:\s*(.+)$', content, re.MULTILINE)
        timestamp = timestamp_match.group(1).strip() if timestamp_match else ""
        
        # Extract audio file path (last line format: "Audio File Path: <path>")
        audio_match = re.search(r'^Audio File Path:\s*(.+)$', content, re.MULTILINE)
        if not audio_match:
            print(f"[BACKUP PARSE] Failed to find audio path in {file_path}")
            return None
        audio_file_path = audio_match.group(1).strip()
        
        # Extract main content (between the separator lines)
        # Format: ========\n\n<content>\n\n========
        content_match = re.search(r'={80}\n\n(.*?)\n\n={80}', content, re.DOTALL)
        if not content_match:
            print(f"[BACKUP PARSE] Failed to find content in {file_path}")
            return None
        main_content = content_match.group(1).strip()
        
        return {
            'url': url,
            'category': category,
            'content': main_content,
            'audio_file_path': audio_file_path,
            'timestamp': timestamp,
            'backup_file': file_path
        }
    except Exception as e:
        print(f"[BACKUP PARSE] Error parsing {file_path}: {e}")
        return None


def list_backup_files() -> List[Path]:
    """
    List all backup files in the backup_content directory.
    
    Returns:
        List of Path objects for .txt files in backup_content/
    """
    backup_dir = Path("backup_content")
    if not backup_dir.exists():
        return []
    
    # Get all .txt files in backup_content/ (not in subdirectories)
    backup_files = [f for f in backup_dir.glob("*.txt") if f.is_file()]
    print(f"[BACKUP LIST] Found {len(backup_files)} backup files")
    return backup_files


def compress_audio(input_path: str, bitrate: str = "64k") -> Optional[str]:
    """
    Compress audio file to MP3 format to reduce file size.
    
    Args:
        input_path: Path to the input audio file
        bitrate: Target bitrate for compression (default: "64k")
                 Options: "32k", "64k", "128k", "192k"
        
    Returns:
        Path to compressed audio file, or None if compression fails
    """
    try:
        from pydub import AudioSegment
        
        input_file = Path(input_path)
        if not input_file.exists():
            print(f"[COMPRESS] Input file not found: {input_path}")
            return None
        
        # Check if file is already small enough
        file_size_mb = input_file.stat().st_size / (1024 * 1024)
        print(f"[COMPRESS] Input file size: {file_size_mb:.2f} MB")
        
        # Create output filename
        output_path = input_file.with_suffix('.mp3')
        # If input is already .mp3, add _compressed suffix
        if input_file.suffix.lower() == '.mp3':
            output_path = input_file.with_stem(f"{input_file.stem}_compressed")
        
        print(f"[COMPRESS] Compressing {input_file.name} → {output_path.name} (bitrate: {bitrate})...")
        
        # Load audio file
        audio = AudioSegment.from_file(str(input_path))
        
        # Export as MP3 with specified bitrate
        audio.export(
            str(output_path),
            format="mp3",
            bitrate=bitrate,
            parameters=["-q:a", "2"]  # Quality setting for VBR (2 = high quality)
        )
        
        # Check compressed file size
        compressed_size_mb = output_path.stat().st_size / (1024 * 1024)
        compression_ratio = (1 - compressed_size_mb / file_size_mb) * 100
        
        print(f"[COMPRESS] ✅ Compression successful!")
        print(f"[COMPRESS]    Original: {file_size_mb:.2f} MB")
        print(f"[COMPRESS]    Compressed: {compressed_size_mb:.2f} MB")
        print(f"[COMPRESS]    Saved: {compression_ratio:.1f}%")
        
        return str(output_path)
        
    except ImportError:
        print("[COMPRESS] ❌ pydub not installed. Install with: pip install pydub")
        print("[COMPRESS] Also requires ffmpeg. Install with: brew install ffmpeg (macOS)")
        return None
    except Exception as e:
        print(f"[COMPRESS] ❌ Error compressing audio: {e}")
        return None
