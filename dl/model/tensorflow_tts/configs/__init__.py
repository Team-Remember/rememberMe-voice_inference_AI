from dl.model.tensorflow_tts.configs.base_config import BaseConfig
from dl.model.tensorflow_tts.configs.fastspeech import FastSpeechConfig
from dl.model.tensorflow_tts.configs.fastspeech2 import FastSpeech2Config
from dl.model.tensorflow_tts.configs.melgan import (
    MelGANDiscriminatorConfig,
    MelGANGeneratorConfig,
)
from dl.model.tensorflow_tts.configs.mb_melgan import (
    MultiBandMelGANDiscriminatorConfig,
    MultiBandMelGANGeneratorConfig,
)
from dl.model.tensorflow_tts.configs.hifigan import (
    HifiGANGeneratorConfig,
    HifiGANDiscriminatorConfig,
)
from dl.model.tensorflow_tts.configs.tacotron2 import Tacotron2Config
from dl.model.tensorflow_tts.configs.parallel_wavegan import ParallelWaveGANGeneratorConfig
from dl.model.tensorflow_tts.configs.parallel_wavegan import ParallelWaveGANDiscriminatorConfig
