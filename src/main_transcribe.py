# main.py
import os
from pathlib import Path
from config import PipelineConfig
from audio_processor import AudioProcessor
from term_corrector import TermCorrector
from transcriber import Transcriber
from subtitle_exporter import SubtitleExporter
import time
import torch

def process_video(video_path: str, config=None):
    print(f"XPU available: {torch.xpu.is_available()}")

    start_time = time.time()

    if config is None:
        config = PipelineConfig()
    
    os.makedirs(config.output_dir, exist_ok=True)
    base_name = Path(video_path).stem

    print("=" * 60)
    print("🎬 Start processing video")
    print(f"Mode: {'Single speaker' if config.one_speaker_flag else 'Two speakers'}")
    print("=" * 60)
    
    # Extract Audio
    audio_processor = AudioProcessor(config)
    audio_path = audio_processor.extract_audio_from_video(
        video_path,
        os.path.join(config.output_dir, f"{base_name}_audio.wav")
    )
    
    # Video Preprocessing
    if config.one_speaker_flag:
        segments = audio_processor.create_audio_segment(audio_path)
    else:
        segments = audio_processor.diarize_with_diarize(audio_path)

        if not segments:
            print("!!! Issue: no segments exist.")
            return
        
        segments = audio_processor.merge_segments(segments, audio_path=audio_path, min_gap=0.5)
    
    # Transcribe by Whisper
    transcriber = Transcriber(config)
    segments = transcriber.transcribe_segments(segments)

    # Rule-based Correct Chinese Terms
    print("term dir: ", config.term_reference_dir)
    if len(config.term_reference_dir) > 1:
        term_corrector = TermCorrector(config)
        segments = term_corrector.correct_segments(segments)
    
    # Cut Segments
    exporter = SubtitleExporter(config)
    
    valid_segments = [s for s in segments if s.get('text', '').strip()]

    valid_segments = exporter.split_and_merge_by_language(
        valid_segments, 
        min_switch_count=3
        )
    
    valid_segments = exporter.split_subtitles_by_clauses(
        valid_segments, 
        min_duration=3.0,
        max_duration=5.0
        )
    
    total_elapsed = time.time() - start_time
    minutes = int(total_elapsed // 60)
    seconds = total_elapsed % 60
    print(f"\nTotal Process Time: {minutes} minutes {seconds:.1f} seconds")

    # Export Subtitles
    if valid_segments:
        srt_path = os.path.join(config.output_dir, f"{base_name}.srt")
        txt_path = os.path.join(config.output_dir, f"{base_name}_with_timestamps.txt")
        
        exporter.export_srt(valid_segments, srt_path)
        exporter.export_txt_with_timestamps(valid_segments, txt_path)
    
    # Clean temporary Files
    if os.path.exists(audio_path):
        os.remove(audio_path)
    
    print(f"\nTranscribe Finished! Output Directory: {config.output_dir}")

def load_config_and_process_transcribe(source_path, output_path, term_file_path, model,
                  use_finetuned, one_speaker, traditional_chinese):

    print(use_finetuned)
    user_overrides = {
        "output_dir":output_path, 
        "term_reference_dir":term_file_path, 
        "whisper_model_name":model,
        "use_finetuned_model":use_finetuned,
        "one_speaker_flag":one_speaker,
        "output_traditional_chinese":traditional_chinese
    }
    config = PipelineConfig()
    config.output_dir = output_path
    user_config = PipelineConfig().update(**user_overrides)
    print("user configuration: ", user_config)
    
    process_video(source_path, user_config)