# transcriber.py
import whisper
import numpy as np
#from pathlib import Path
from typing import List, Dict, Optional
#import re
import torch

class Transcriber:
    def __init__(self, config):
        self.config = config

        self._load_original_model()
        
        self.dummy_text = "(Cannot detect, please add manually/無法偵測，需手動添加)"
        self.base_transcribe_kwargs = {
            "task": "transcribe",
            "temperature": [0.0, 0.2, 0.4],
            "compression_ratio_threshold": 1.5,
            "logprob_threshold": -0.8,
            "no_speech_threshold": 0.7,
        }

        self.openAI_extra = {
            "condition_on_previous_text": False,
            "verbose": False,
            "fp16": False,
            "word_timestamps": False,     # 对应 HF 的 return_timestamps
            "beam_size": 10,              # 对应 HF 的 num_beams
            "best_of": 10,                # 需配合 temperature>0
        }


    def _load_original_model(self):
        print(f"Loading original Whisper model: {self.config.whisper_model_name}...")
        self.model = whisper.load_model(self.config.whisper_model_name)
        if torch.xpu.is_available():
            self.model = self.model.to('xpu')
            print("Use Intel XPU")
        elif torch.cuda.is_available():
            self.model = self.model.to('cuda')
            print("Use CUDA")
        else:
            print("Use CPU")
        self._is_finetuned = False
    
    def _transcribe_original(self, audio_float: np.ndarray) -> Dict:
        """
            Transcribe using original Whisper model
            Args:
                audio_float: a piece of audio for transcribe task       
            Returns:
                Dict: Whisper output
        """
        transcribe_kwargs = {
            'audio': audio_float,
            'verbose': False,
            'fp16': False,
            'task': 'transcribe',
            'temperature': [0.0, 0.2, 0.4],
            'compression_ratio_threshold': 1.5,
            'logprob_threshold': -0.8,
            'no_speech_threshold': 0.7,
            'condition_on_previous_text': False,
            'word_timestamps': False,
            'beam_size': 10,
            'best_of': 10,
        }

        try:
            result = self.model.transcribe(**transcribe_kwargs)
            return result
        except Exception as e:
            print(f"Transcribe failed: {e}")
            return {'segments': [], 'text': ''}

    def transcribe_segment(self, audio_data: np.ndarray) -> Dict:
        """
            Transcribe single audio segment
            Args:
                audio_data: a piece of audio       
            Returns:
                Dict: Whisper output
        """
        if audio_data.dtype == np.int16:
            audio_float = audio_data.astype(np.float32) / 32768.0
        else:
            audio_float = audio_data.astype(np.float32)

        return self._transcribe_original(audio_float)
            
    def transcribe_segments(self, segments: List[Dict]) -> List[Dict]:
        """
            Transcribe a list of audio segments
            Args:
                segments: a  list of audio segments      
            Returns:
                List[Dict]: a list of Whisper outputs for all segments
        """
        print(f"Transcribe {len(segments)} segments...")
        
        success_count = 0
        total_text_length = 0
        dummy_count = 0
        
        for idx, segment in enumerate(segments):
            audio = segment.get('audio')
            if audio is None or len(audio) == 0:
                segment['text'] = self.dummy_text
                segment['transcription_success'] = False
                segment['is_dummy'] = True
                dummy_count += 1
                print(f"Audio Piece {idx+1}: is blank, use dummy text")
                continue
            
            print(f"  Audio piece {idx+1}/{len(segments)} "
                  f"{segment['start']:.2f}s-{segment['end']:.2f}s)")
            
            try:
                result = self.transcribe_segment(audio)
                
                texts = [seg['text'].strip() for seg in result.get('segments', [])]
                full_text = ' '.join(texts)
                
                if full_text.strip():
                    segment['text'] = full_text
                    success_count += 1
                    total_text_length += len(full_text)
                    preview = full_text[:50].replace('\n', ' ')
                    print(f"Transcript preview: {preview}...")
                else:
                    segment['text'] = self.dummy_text
                    dummy_count += 1
                    print(f"No transcript, use dummy text")
                    
            except Exception as e:
                print(f"Transcribe failed: {e}, use dummy text")
                segment['text'] = self.dummy_text
                dummy_count += 1
        
        print(f"\n📊 Transcribe task matrices:")
        print(f"Success pieces: {success_count}/{len(segments)} ({success_count/len(segments)*100:.1f}%)")
        print(f"Dummy pieces: {dummy_count} ({dummy_count/len(segments)*100:.1f}%)")
        print(f"Transcribe completed.")
        
        return segments