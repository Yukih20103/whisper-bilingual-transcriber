# subtitle_exporter.py
from typing import List, Dict
from phrasplit import split_long_lines
from zhconv import convert
import jieba
import pyphen
import re

pyphen_en = pyphen.Pyphen(lang="en_US")


def _detect_lang(token: str):
    """Return 'zh', 'en', or None."""
    if re.search(r"[\u4e00-\u9fff]", token):
        return "zh"
    if re.search(r"[a-zA-Z]", token):
        return "en"
    return None


def _count_syllables_en(word: str) -> int:
    """Approximate English syllable count using pyphen."""
    word = word.lower().strip("'")
    if not word:
        return 0
    hyphenated = pyphen_en.inserted(word)
    if hyphenated:
        return len(hyphenated.split("-"))
    groups = re.findall(r"[aeiouy]+", word)
    return max(1, len(groups))


def _sound_count(token: str) -> int:
    """
    Count syllables.
    Chinese: each Han character = 1 syllable.
    English: estimated syllables per word.
    Punctuation ignored.
    """
    han_count = len(re.findall(r"[\u4e00-\u9fff]", token))
    latin_words = re.findall(r"[a-zA-Z']+", token)
    en_count = sum(_count_syllables_en(w) for w in latin_words)
    return han_count + en_count


def _tokenize_mixed(text: str):
    """
    Tokenize mixed Chinese/English text.
    Chinese runs -> jieba words.
    English runs -> words.
    Spaces/punctuation -> attached to previous token.
    """
    raw_parts = re.findall(
        r"[\u4e00-\u9fff]+|[a-zA-Z']+|[^a-zA-Z\u4e00-\u9fff]+",
        text,
    )
    tokens = []
    for part in raw_parts:
        if re.fullmatch(r"[\u4e00-\u9fff]+", part):
            for word in jieba.lcut(part):
                if word:
                    tokens.append(word)
        elif re.fullmatch(r"[a-zA-Z']+", part):
            tokens.append(part)
        else:
            if tokens:
                tokens[-1] += part
            else:
                tokens.append(part)
    return tokens

class SubtitleExporter:
    def __init__(self, config=None):
        self.config = config 
    
    @staticmethod
    def format_timestamp(seconds: float) -> str:
        """Format timestamps as SRT format"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
    
    def _convert_text(self, text: str) -> str:
        """Convert simplified Chinese to Traditional Chinese character"""
        if not text:
            return text
        
        if self.config and getattr(self.config, 'output_traditional_chinese', False):
            try:
                converted_text = convert(text, 'zh-tw')
                return converted_text
            except:
                return text
        return text

    def split_and_merge_by_language(self, segments, min_switch_count: int = 3) -> List[Dict]:
        """
        Split mixed-language (zh/en) segments by language boundaries,
        then merge adjacent same-language segments.

        min_switch_count:
            Minimum number of WORDS in a different language before
            a switch is confirmed.
        """
        new_segments = []

        for seg in segments:
            text = seg.get("text", "").strip()
            if not text:
                continue

            tokens = _tokenize_mixed(text)

            main_is_en = None # True = en, False = zh
            main_buf = []
            temp_buf = []
            temp_count = 0
            groups = []   # list of (lang_str, tokens)

            for token in tokens:
                lang = _detect_lang(token)

                # punctuation / spaces attach to previous token
                if lang is None:
                    if temp_buf:
                        temp_buf[-1] += token
                    elif main_buf:
                        main_buf[-1] += token
                    continue

                token_is_en = (lang == "en")

                # first token
                if main_is_en is None:
                    main_is_en = token_is_en
                    main_buf.append(token)
                    continue

                if token_is_en == main_is_en:
                    if temp_buf:
                        main_buf.extend(temp_buf)
                        temp_buf = []
                        temp_count = 0
                    main_buf.append(token)

                else:
                    # different from main
                    temp_buf.append(token)
                    temp_count += 1

                    # confirm switch when greater than or equal to threshold
                    if temp_count >= min_switch_count:
                        groups.append(
                            ("en" if main_is_en else "zh", main_buf)
                        )
                        main_buf = temp_buf
                        main_is_en = not main_is_en
                        temp_buf = []
                        temp_count = 0

            if temp_buf:
                main_buf.extend(temp_buf)

            if main_buf:
                groups.append(
                    ("en" if main_is_en else "zh", main_buf)
                )

            #time allocation by syllables
            group_sounds = [
                sum(_sound_count(t) for t in group_tokens)
                for _, group_tokens in groups
            ]
            total_sounds = sum(group_sounds)
            if total_sounds == 0:
                continue

            current_sound = 0

            for (lang, group_tokens), g_sound in zip(groups, group_sounds):
                group_text = "".join(group_tokens).strip()
                if not group_text:
                    continue

                start_ratio = current_sound / total_sounds
                end_ratio = (current_sound + g_sound) / total_sounds
                current_sound += g_sound

                group_start = seg["start"] + start_ratio * (
                    seg["end"] - seg["start"]
                )
                group_end = seg["start"] + end_ratio * (
                    seg["end"] - seg["start"]
                )

                new_segments.append(
                    {
                        "start": group_start,
                        "end": group_end,
                        "text": group_text,
                        "language": lang,
                        "speaker": seg.get("speaker", "unknown"),
                    }
                )

        # merge adjacent same-language segments
        merged_segments = []

        for seg in new_segments:
            if not merged_segments:
                merged_segments.append(seg)
                continue

            last = merged_segments[-1]

            if seg.get("language") == last.get("language"):
                sep = (
                    ""
                    if last["text"].endswith(("。", "！", "？", "，", "；", "："))
                    else " "
                )
                last["text"] = last["text"] + sep + seg["text"]
                last["end"] = seg["end"]
                last["duration"] = last["end"] - last["start"]
            else:
                merged_segments.append(seg)

        print(
            f"  ✂️ Cut segments by language: "
            f"{len(segments)} → {len(merged_segments)} segments"
        )
        return merged_segments

    def split_subtitles_by_clauses(self, segments: List[Dict], min_duration: float = 3.0, max_duration: float = 5.0) -> List[Dict]:
        """
        按语义切分字幕，并确保每个片段时长在 [min_duration, max_duration] 之间
        (尽可能满足，若无法满足则保留短片段)
        """
        # ---- 第一步：原始切割逻辑（保持不变） ----
        new_segments = []
        for seg in segments:
            text = seg.get('text', '').strip()
            start = seg['start']
            end = seg['end']
            duration = end - start

            if not text:
                continue

            if duration <= max_duration:
                new_segments.append(seg.copy())
                continue

            # 根据语言设定最大字符数
            if seg.get('language') == "en":
                MAX_CHARS = 50
            else:
                MAX_CHARS = 15

            parts = split_long_lines(text, max_length=MAX_CHARS)
            parts = [p.strip() for p in parts if p.strip()]

            if len(parts) <= 1:
                new_segments.append(seg.copy())
                continue

            # 按字符比例分配时间
            char_counts = [len(p) for p in parts]
            total_chars = sum(char_counts)
            current_time = start

            for part, char_count in zip(parts, char_counts):
                time_ratio = char_count / total_chars
                part_duration = duration * time_ratio

                part_start = current_time
                part_end = min(part_start + part_duration, end)

                new_segments.append({
                    'start': part_start,
                    'end': part_end,
                    'text': part,
                    'language': seg.get('language', 'unknown'),
                    'speaker': seg.get('speaker', 'unknown'),
                })
                current_time = part_end

        merged_segments = []
        i = 0
        while i < len(new_segments):
            current = new_segments[i].copy()

            while i + 1 < len(new_segments):
                next_seg = new_segments[i + 1]
                if current['end'] - current['start'] >= min_duration:
                    break
                if next_seg.get('language') != current.get('language'):
                    break
                
                current['end'] = next_seg['end']
                current['text'] = current['text'] + ' ' + next_seg['text']
                i += 1  
            merged_segments.append(current)
            i += 1

        print(f"  ✅ Cut by clauses finished: {len(segments)} → {len(merged_segments)} segments")
        return merged_segments

    def export_srt(self, transcripts: List[Dict], output_path: str):
        """Export SRT subtitle files"""
        with open(output_path, 'w', encoding='utf-8') as f:
            idx = 1
            for seg in transcripts:
                if seg.get('is_dummy', False):
                    continue
            
                start = self.format_timestamp(seg['start'])
                end = self.format_timestamp(seg['end'])
                text = self._convert_text(seg['text'])
                f.write(f"{idx}\n")
                f.write(f"{start} --> {end}\n")
                f.write(f"{text}\n\n")
                idx += 1
    
    def export_txt_with_timestamps(self, transcripts: List[Dict], output_path: str):
        """Export TXT files with timestamps"""
        with open(output_path, 'w', encoding='utf-8') as f:
            for seg in transcripts:
                if seg.get('is_dummy', False):
                    continue
            
                start = self.format_timestamp(seg['start'])
                end = self.format_timestamp(seg['end'])
                text = self._convert_text(seg['text'])
                f.write(f"[{start} --> {end}] {text}\n")