import subprocess
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import List, Dict
from diarize import diarize

class AudioProcessor:
    
    def __init__(self, config):
        self.config = config
        self.encoder = None
        self.speaker_references = {}

    def create_audio_segment(self, audio_path: str) -> List[Dict]:
        """
        Assign the language to segment (single language)
        Args:
            audio_path: path of audio
            
        Returns:
            List[Dict]: a video segment with metadata
        """
    
        # read the audio from audio_path
        audio_data, sample_rate = sf.read(audio_path, dtype='int16')
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1).astype(np.int16)
    
        total_duration = len(audio_data) / sample_rate
    
        segments = [{
            'speaker': 'speaker_00',
            'start': 0.0,
            'end': total_duration,
            'audio': audio_data,
            'duration': total_duration,
        }]
        return segments

    def extract_audio_from_video(self, input_path: str, audio_path: str = None) -> str:
        """
            Extract audio from the video and save it to a audio path
            Args:
                input_path: path of a video
                audio_path: output path of the extracted audio
                    
            Returns:
                str: a audio path extracted from the video
        """
        if audio_path is None:
            audio_path = Path(input_path).stem + '_audio.wav'
            input_dir = Path(input_path).parent
            audio_path = str(input_dir / (Path(input_path).stem + '_audio.wav'))

        print(f"Extracting audio: {Path(input_path).name} -> {Path(audio_path).name}")

        cmd = [
            'ffmpeg', '-i', input_path,
            '-acodec', 'pcm_s16le',
            '-ar', str(self.config.sample_rate),
            '-ac', '1',
            '-y', audio_path
        ]
    
        subprocess.run(cmd, capture_output=True, check=True)
        info = sf.info(audio_path)
        print(f" Sample size: {info.samplerate} Hz, Channels: {info.channels}")
        return audio_path

    def diarize_with_diarize(self, audio_path: str) -> List[Dict]:
        """
            Diarize audio by speaker
            Args:
                audio_path: path of the extracted audio
                            
            Returns:
                List[Dict]: a list of audio data of video segments
        """
        print(f"🎤 说话人分段 (diarize): {Path(audio_path).name}")
        
        result = diarize(audio_path, num_speakers=2)
        
        segments = []
        for seg in result.segments:
            segments.append({
                'speaker': seg.speaker,
                'start': seg.start,
                'end': seg.end,
                'duration': seg.duration
            })
        
        print(f"Detected {len(segments)} segments")
        return segments
    
    def merge_segments(self, segments: List[Dict], audio_path: str, min_gap: float = 0.5) -> List[Dict]:
        """
        Merge successive segments with the same speaker to preserve context
        Args:
            segments: metadata of audio segements
            audio_path: path of the extracted audio
            min_gap: minimum gap between consecutive segements
    
        Returns:
            List[Dict]: a list of merged segments with audio pieces
        """
        if not segments:
            return []
        print(f"  Merge segments: {len(segments)}  to ", end='')
        audio_data, sample_rate = sf.read(audio_path, dtype='int16')
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1).astype(np.int16)
    
        sorted_segments = sorted(segments, key=lambda x: x['start'])
    
        merged = []
        current = sorted_segments[0].copy()
    
        for next_seg in sorted_segments[1:]:
            time_gap = next_seg['start'] - current['end']
        
            # merge if time gap is smaller than minimum gap and speakers of 2 segments are the same
            if (next_seg['speaker'] == current['speaker'] and time_gap < min_gap):
                current['end'] = max(current['end'], next_seg['end'])
                current['duration'] = current['end'] - current['start']
            else:
                start_time = int(current['start'] * sample_rate)
                end_time = int(current['end'] * sample_rate)
                start_time = max(0, min(start_time, len(audio_data)))
                end_time = max(0, min(end_time, len(audio_data)))         
                                
                current['audio'] = audio_data[start_time:end_time]
                merged.append(current)
                current = next_seg.copy()
    
        # add the final segment
        start_time = int(current['start'] * sample_rate)
        end_time = int(current['end'] * sample_rate)
        start_time = max(0, min(start_time, len(audio_data) - 1))
        end_time = max(0, min(end_time, len(audio_data)))         
                                        
        current['audio'] = audio_data[start_time:end_time]
        merged.append(current)
    
        # filter segments shorter than minimum duration
        min_duration = getattr(self.config, 'min_segment_duration', 0.5)
        merged = [seg for seg in merged if seg['duration'] >= min_duration]
    
        print(f"{len(merged)}")
        return merged