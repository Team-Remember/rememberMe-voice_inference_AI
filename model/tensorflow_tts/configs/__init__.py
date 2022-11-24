from model.tensorflow_tts.configs.base_config import BaseConfig
from model.tensorflow_tts.configs.fastspeech import FastSpeechConfig
from model.tensorflow_tts.configs.fastspeech2 import FastSpeech2Config
from model.tensorflow_tts.configs.melgan import (
    MelGANDiscriminatorConfig,
    MelGANGeneratorConfig,
)
from model.tensorflow_tts.configs.mb_melgan import (
    MultiBandMelGANDiscriminatorConfig,
    MultiBandMelGANGeneratorConfig,
)
from model.tensorflow_tts.configs.hifigan import (
    HifiGANGeneratorConfig,
    HifiGANDiscriminatorConfig,
)
from model.tensorflow_tts.configs.tacotron2 import Tacotron2Config
from model.tensorflow_tts.configs.parallel_wavegan import ParallelWaveGANGeneratorConfig
from model.tensorflow_tts.configs.parallel_wavegan import ParallelWaveGANDiscriminatorConfig
