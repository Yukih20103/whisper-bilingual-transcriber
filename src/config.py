# config.py
class PipelineConfig:
    def __init__(self):
        # Whisper Configure
        self.whisper_model_name = "large-v3"  # tiny/base/small/medium/large
        self.output_dir = "./output"
        self.sample_rate = 16000

        self.term_reference_dir = ''

        # Speaker Configure
        self.one_speaker_flag = False
        self.output_traditional_chinese = True 
        self.use_finetuned_model = False

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if not hasattr(self, key):
                raise AttributeError(f"Unknown config key: {key}")
            setattr(self, key, value)
        return self

    def __repr__(self):
        attrs = "\n".join(f"  {k} = {v!r}" for k, v in self.__dict__.items())
        return f"{self.__class__.__name__}(\n{attrs}\n)"

    __str__ = __repr__