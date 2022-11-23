from dl.model.tensorflow_tts.models.base_model import BaseModel
from dl.model.tensorflow_tts.models.fastspeech import TFFastSpeech
from dl.model.tensorflow_tts.models.fastspeech2 import TFFastSpeech2
from dl.model.tensorflow_tts.models.melgan import (
    TFMelGANDiscriminator,
    TFMelGANGenerator,
    TFMelGANMultiScaleDiscriminator,
)
from dl.model.tensorflow_tts.models.mb_melgan import TFPQMF
from dl.model.tensorflow_tts.models.mb_melgan import TFMBMelGANGenerator
from dl.model.tensorflow_tts.models.hifigan import (
    TFHifiGANGenerator,
    TFHifiGANMultiPeriodDiscriminator,
    TFHifiGANPeriodDiscriminator
)
from dl.model.tensorflow_tts.models.tacotron2 import TFTacotron2
from dl.model.tensorflow_tts.models.parallel_wavegan import TFParallelWaveGANGenerator
from dl.model.tensorflow_tts.models.parallel_wavegan import TFParallelWaveGANDiscriminator
