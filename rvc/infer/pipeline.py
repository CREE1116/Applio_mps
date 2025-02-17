
import os
import gc
import re
import sys
import torch
import torch.nn.functional as F
import torchcrepe
import faiss
import librosa
import numpy as np
from scipy import signal
from torch import Tensor
import traceback  # traceback import 추가

now_dir = os.getcwd()
sys.path.append(now_dir)

from rvc.lib.predictors.RMVPE import RMVPE0Predictor
from rvc.lib.predictors.FCPE import FCPEF0Predictor

import logging

logging.getLogger("faiss").setLevel(logging.WARNING)

FILTER_ORDER = 5
CUTOFF_FREQUENCY = 48  # Hz
SAMPLE_RATE = 16000  # Hz
bh, ah = signal.butter(
    N=FILTER_ORDER, Wn=CUTOFF_FREQUENCY, btype="high", fs=SAMPLE_RATE
)

input_audio_path2wav = {}


class AudioProcessor:
    """
    A class for processing audio signals, specifically for adjusting RMS levels.
    """

    def change_rms(
        source_audio: np.ndarray,
        source_rate: int,
        target_audio: np.ndarray,
        target_rate: int,
        rate: float,
    ):
        """
        Adjust the RMS level of target_audio to match the RMS of source_audio, with a given blending rate.

        Args:
            source_audio: The source audio signal as a NumPy array.
            source_rate: The sampling rate of the source audio.
            target_audio: The target audio signal to adjust.
            target_rate: The sampling rate of the target audio.
            rate: The blending rate between the source and target RMS levels.
        """
        # Calculate RMS of both audio data
        rms1 = librosa.feature.rms(
            y=source_audio,
            frame_length=source_rate // 2 * 2,
            hop_length=source_rate // 2,
        )
        rms2 = librosa.feature.rms(
            y=target_audio,
            frame_length=target_rate // 2 * 2,
            hop_length=target_rate // 2,
        )

        # Interpolate RMS to match target audio length
        rms1 = F.interpolate(
            torch.from_numpy(rms1).float().unsqueeze(0),
            size=target_audio.shape[0],
            mode="linear",
        ).squeeze()
        rms2 = F.interpolate(
            torch.from_numpy(rms2).float().unsqueeze(0),
            size=target_audio.shape[0],
            mode="linear",
        ).squeeze()
        rms2 = torch.maximum(rms2, torch.zeros_like(rms2) + 1e-6)

        # Adjust target audio RMS based on the source audio RMS
        adjusted_audio = (
            target_audio
            * (torch.pow(rms1, 1 - rate) * torch.pow(rms2, rate - 1)).numpy()
        )
        return adjusted_audio



class Autotune:
    """
    A class for applying autotune to a given fundamental frequency (F0) contour.
    """

    def __init__(self, ref_freqs):
        """
        Initializes the Autotune class with a set of reference frequencies.

        Args:
            ref_freqs: A list of reference frequencies representing musical notes.
        """
        self.ref_freqs = ref_freqs
        self.note_dict = self.ref_freqs  # No interpolation needed

    def autotune_f0(self, f0, f0_autotune_strength):
        """
        Autotunes a given F0 contour by snapping each frequency to the closest reference frequency.

        Args:
            f0: The input F0 contour as a NumPy array.
        """
        autotuned_f0 = np.zeros_like(f0)
        for i, freq in enumerate(f0):
            closest_note = min(self.note_dict, key=lambda x: abs(x - freq))
            autotuned_f0[i] = freq + (closest_note - freq) * f0_autotune_strength
        return autotuned_f0


class Pipeline:
    # ... (get_f0_crepe, get_f0_hybrid, get_f0 내용은 이전 수정본과 동일)
    def __init__(self, tgt_sr, config):
        """Initializes the Pipeline class with window size."""
        self.x_pad = config.x_pad
        self.x_query = config.x_query
        self.x_center = config.x_center
        self.x_max = config.x_max
        self.sample_rate = 16000
        self.window = 160  # window size 속성 추가 (해결!)
        self.t_pad = self.sample_rate * self.x_pad
        self.t_pad_tgt = tgt_sr * self.x_pad
        self.t_pad2 = self.t_pad * 2
        self.t_query = self.sample_rate * self.x_query
        self.t_center = self.sample_rate * self.x_center
        self.t_max = self.sample_rate * self.x_max
        self.time_step = self.window / self.sample_rate * 1000
        self.f0_min = 50
        self.f0_max = 1100
        self.f0_mel_min = 1127 * np.log(1 + self.f0_min / 700)
        self.f0_mel_max = 1127 * np.log(1 + self.f0_max / 700)
        self.device = config.device
        print(f"Pipeline 초기화: device = {self.device}")  # Pipeline device 확인
        self.ref_freqs = [
            49.00,  # G1
            51.91,  # G#1 / Ab1
            55.00,  # A1
            58.27,  # A#1 / Bb1
            61.74,  # B1
            65.41,  # C2
            69.30,  # C#2 / Db2
            73.42,  # D2
            77.78,  # D#2 / Eb2
            82.41,  # E2
            87.31,  # F2
            92.50,  # F#2 / Gb2
            98.00,  # G2
            103.83,  # G#2 / Ab2
            110.00,  # A2
            116.54,  # A#2 / Bb2
            123.47,  # B2
            130.81,  # C3
            138.59,  # C#3 / Db3
            146.83,  # D3
            155.56,  # D#3 / Eb3
            164.81,  # E3
            174.61,  # F3
            185.00,  # F#3 / Gb3
            196.00,  # G3
            207.65,  # G#3 / Ab3
            220.00,  # A3
            233.08,  # A#3 / Bb3
            246.94,  # B3
            261.63,  # C4
            277.18,  # C#4 / Db4
            293.66,  # D4
            311.13,  # D#4 / Eb4
            329.63,  # E4
            349.23,  # F4
            369.99,  # F#4 / Gb4
            392.00,  # G4
            415.30,  # G#4 / Ab4
            440.00,  # A4
            466.16,  # A#4 / Bb4
            493.88,  # B4
            523.25,  # C5
            554.37,  # C#5 / Db5
            587.33,  # D5
            622.25,  # D#5 / Eb5
            659.25,  # E5
            698.46,  # F5
            739.99,  # F#5 / Gb5
            783.99,  # G5
            830.61,  # G#5 / Ab5
            880.00,  # A5
            932.33,  # A#5 / Bb5
            987.77,  # B5
            1046.50,  # C6
        ]
        self.autotune = Autotune(self.ref_freqs)
        self.note_dict = self.autotune.note_dict
        # Initialize RMVPE, but might not be used if f0_method is None
        self.model_rmvpe = RMVPE0Predictor(
            os.path.join("rvc", "models", "predictors", "rmvpe.pt"),
            device=self.device,
        )
        print(f"RMVPE 모델 로드 완료, device: {self.device}")  # RMVPE 모델 device 확인

        self.f0_method = None  # Default to None for no F0 estimation

    def get_f0_crepe(
        self,
        x,
        f0_min,
        f0_max,
        p_len,
        hop_length,
        model="full",
    ):
        """
        Estimates the fundamental frequency (F0) of a given audio signal using the Crepe model.
        """
        x = x.astype(np.float32)
        x /= np.quantile(np.abs(x), 0.999)
        audio = torch.from_numpy(x).to(self.device, copy=True)
        print(f"Crepe 입력 오디오 텐서 device: {audio.device}")  # Crepe 입력 텐서 device 확인
        audio = torch.unsqueeze(audio, dim=0)
        if audio.ndim == 2 and audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)  # .detach() 제거: 불필요한 detach 제거
        audio = audio.detach()
        pitch: Tensor = torchcrepe.predict(
            audio,
            self.sample_rate,
            hop_length,
            f0_min,
            f0_max,
            model,
            batch_size=hop_length * 2,
            device=self.device,
            pad=True,
        )
        print(f"Crepe 출력 pitch 텐서 device: {pitch.device}")  # Crepe 출력 텐서 device 확인
        p_len = p_len or x.shape[0] // hop_length
        source = np.array(pitch.squeeze(0).cpu().float().numpy())  # pitch tensor를 numpy array로 변환 (기존 코드 유지)
        source[source < 0.001] = np.nan
        target = np.interp(
            np.arange(0, len(source) * p_len, len(source)) / p_len,
            np.arange(0, len(source)),
            source,
        )  # f0 interpolation (기존 코드 유지)
        f0 = np.nan_to_num(target)  # NaN 값을 0으로 채우기 (기존 코드 유지)
        return f0

    def get_f0_hybrid(
        self,
        methods_str,
        x,
        f0_min,
        f0_max,
        p_len,
        hop_length,
    ):
        """
        Estimates the fundamental frequency (F0) using a hybrid approach combining multiple methods.
        """
        methods_str = re.search("hybrid\[(.+)\]", methods_str)
        if methods_str:
            methods = [method.strip() for method in methods_str.group(1).split("+")]
        f0_computation_stack = []
        print(f"Calculating f0 pitch estimations for methods: {', '.join(methods)}")
        x = x.astype(np.float32)
        x /= np.quantile(np.abs(x), 0.999)
        for method in methods:
            f0 = None
            if method == "crepe":
                f0 = self.get_f0_crepe_computation(
                    x, f0_min, f0_max, p_len, int(hop_length)
                )
            elif method == "rmvpe":
                f0 = self.model_rmvpe.infer_from_audio(x, thred=0.03)
                print(f"RMVPE F0 device: {f0.device if isinstance(f0, torch.Tensor) else 'NumPy Array'}")  # RMVPE F0 device 확인
                f0 = f0[1:]
            elif method == "fcpe":
                self.model_fcpe = FCPEF0Predictor(  # FCPEF0Predictor 객체 생성 (기존 코드 유지)
                    os.path.join("rvc", "models", "predictors", "fcpe.pt"),
                    f0_min=int(f0_min),
                    f0_max=int(f0_max),
                    dtype=torch.float32,
                    device=self.device,
                    sample_rate=self.sample_rate,
                    threshold=0.03,
                )
                print(f"FCPE 모델 로드 완료, device: {self.model_fcpe.device}")  # FCPE 모델 device 확인
                f0 = self.model_fcpe.compute_f0(x, p_len=p_len)  # FCPEF0Predictor를 사용하여 F0 계산 (기존 코드 유지)
                print(f"FCPE F0 device: {f0.device}")  # FCPE F0 device 확인
                del self.model_fcpe
                gc.collect()
            f0_computation_stack.append(f0)

        f0_computation_stack = [fc for fc in f0_computation_stack if fc is not None]
        f0_median_hybrid = None
        if len(f0_computation_stack) == 1:
            f0_median_hybrid = f0_computation_stack[0]
        else:
            f0_median_hybrid = np.nanmedian(f0_computation_stack, axis=0)
        return f0_median_hybrid

    def get_f0(
        self,
        input_audio_path,
        x,
        p_len,
        pitch,
        f0_method,
        hop_length,
        f0_autotune,
        f0_autotune_strength,
        inp_f0=None,
    ):
        """
        Estimates the fundamental frequency (F0) of a given audio signal using various methods.
        """
        global input_audio_path2wav
        f0 = None  # Initialize f0 to None, will be used if f0_method is None
        if f0_method == "crepe":
            f0 = self.get_f0_crepe(x, self.f0_min, self.f0_max, p_len, int(hop_length))
        elif f0_method == "crepe-tiny":
            f0 = self.get_f0_crepe(
                x, self.f0_min, self.f0_max, p_len, int(hop_length), "tiny"
            )
        elif f0_method == "rmvpe":
            f0 = self.model_rmvpe.infer_from_audio(x, thred=0.03)
            print(f"RMVPE F0 device: {f0.device if isinstance(f0, torch.Tensor) else 'NumPy Array'}")  # RMVPE F0 device 확인
        elif f0_method == "fcpe":
            self.model_fcpe = FCPEF0Predictor(
                os.path.join("rvc", "models", "predictors", "fcpe.pt"),
                f0_min=int(self.f0_min),
                f0_max=int(self.f0_max),
                dtype=torch.float32,
                device=self.device,
                sample_rate=self.sample_rate,
                threshold=0.03,
            )
            print(f"FCPE 모델 로드 완료, device: {self.model_fcpe.device}")  # FCPE 모델 device 확인
            f0 = self.model_fcpe.compute_f0(x, p_len=p_len)
            print(f"FCPE F0 device: {f0.device}")  # FCPE F0 device 확인
            del self.model_fcpe
            gc.collect()
        elif "hybrid" in f0_method:
            input_audio_path2wav[input_audio_path] = x.astype(np.double)
            f0 = self.get_f0_hybrid(
                f0_method,
                x,
                self.f0_min,
                self.f0_max,
                p_len,
                hop_length,
            )

        if f0 is not None:  # Only apply autotune, pitch shift, and mel conversion if f0 was actually computed
            if f0_autotune is True:  # F0 autotune 활성화 시 (기존 코드 유지)
                f0 = Autotune.autotune_f0(self, f0, f0_autotune_strength)

            f0 *= pow(2, pitch / 12)  # pitch shift 적용 (기존 코드 유지)
            tf0 = self.sample_rate // self.window
            if inp_f0 is not None:  # 외부 F0 파일 입력 시 (기존 코드 유지)
                delta_t = np.round(
                    (inp_f0[:, 0].max() - inp_f0[:, 0].min()) * tf0 + 1
                ).astype("int16")
                replace_f0 = np.interp(
                    list(range(delta_t)), inp_f0[:, 0] * 100, inp_f0[:, 1]
                )  # F0 interpolation (기존 코드 유지)
                shape = f0[self.x_pad * tf0 : self.x_pad * tf0 + len(replace_f0)].shape[0]
                f0[self.x_pad * tf0 : self.x_pad * tf0 + len(replace_f0)] = replace_f0[
                    :shape
                ]
            f0bak = f0.copy()
            f0_mel = 1127 * np.log(1 + f0 / 700)
            f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - self.f0_mel_min) * 254 / (
                self.f0_mel_max - self.f0_mel_min  # Mel scale normalization (기존 코드 유지)
            ) + 1
            f0_mel[f0_mel <= 1] = 1
            f0_mel[f0_mel > 255] = 255
            f0_coarse = np.rint(f0_mel).astype(int)
            return f0_coarse, f0bak  # F0 coarse, F0 original 반환 (기존 코드 유지)
        else:  # f0_method is None, so return None for both f0_coarse and f0bak
            return None, None

    def voice_conversion(
        self,
        model,
        net_g,
        sid,
        audio0,
        pitch,
        pitchf,
        index,
        big_npy,
        index_rate,
        version,
        protect,
    ):
        """Performs voice conversion with detailed logging."""
        try:  # try-except 블록으로 감싸기
            with torch.no_grad():
                pitch_guidance = pitch is not None and pitchf is not None
                # prepare source audio
                feats = torch.from_numpy(audio0).float()
                feats = feats.mean(-1) if feats.dim() == 2 else feats
                assert feats.dim() == 1, feats.dim()
                feats = feats.view(1, -1).to(self.device)
                print(f"[VC START] Feature extractor 입력 텐서 device: {feats.device}")  # 로그 추가
                # extract features
                feats = model(feats)["last_hidden_state"]
                print(f"[VC FEATS EXTRACTED] Feature extractor 출력 텐서 device: {feats.device}")  # 로그 추가
                feats = (  # v1 모델에 final_proj 레이어 적용
                    model.final_proj(feats[0]).unsqueeze(0) if version == "v1" else feats
                )
                print(f"[VC FINAL PROJ] Final proj 레이어 후 텐서 device: {feats.device}")  # 로그 추가
                # make a copy for pitch guidance and protection
                feats0 = feats.clone() if pitch_guidance else None
                if index is not None and index_rate > 0:
                    print("[VC INDEX RETRIEVAL START]")  # 로그 추가
                    feats = self._retrieve_speaker_embeddings(
                        feats, index, big_npy, index_rate
                    )
                    print(
                        f"[VC INDEX RETRIEVAL END] Speaker embedding retrieval 후 텐서 device: {feats.device}"
                    )  # 로그 추가
                # feature upsampling
                print("[VC INTERPOLATION START]")  # 로그 추가
                feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(
                    0, 2, 1
                )
                print(f"[VC INTERPOLATION END] Feature interpolation 후 텐서 device: {feats.device}")  # 로그 추가
                # adjust the length if the audio is short
                p_len = min(audio0.shape[0] // self.window, feats.shape[1])
                if pitch_guidance:
                    feats0 = F.interpolate(feats0.permute(0, 2, 1), scale_factor=2).permute(
                        0, 2, 1
                    )
                    if pitch is not None and pitchf is not None:
                        pitch, pitchf = pitch[:, :p_len], pitchf[:, :p_len]
                        # Pitch protection blending
                        if protect < 0.5:
                            pitchff = pitchf.clone()
                            pitchff[pitchf > 0] = 1
                            pitchff[pitchf < 1] = protect
                            feats = feats * pitchff.unsqueeze(-1) + feats0 * (
                                1 - pitchff.unsqueeze(-1)
                            )
                            feats = feats.to(feats0.dtype)
                    else:
                        pitch_guidance = False
                else:
                    pitch, pitchf = None, None

                p_len = torch.tensor([p_len], device=self.device).long()
                print(f"[VC GENERATOR INFER START] Generator 입력 feats 텐서 device: {feats.device}")  # 로그 추가
                print(
                    f"[VC GENERATOR INFER START] Generator 입력 pitch 텐서 device: {pitch.device if pitch is not None else 'None'}"
                )  # 로그 추가
                print(
                    f"[VC GENERATOR INFER START] Generator 입력 pitchf 텐서 device: {pitchf.device if pitchf is not None else 'None'}"
                )  # 로그 추가
                audio1 = (  # net_g.infer 호출하여 음성 합성
                    (net_g.infer(feats.float(), p_len, pitch, pitchf.float(), sid)[0][0, 0])
                    .data.cpu()
                    .float()
                    .numpy()
                )
                print("[VC GENERATOR INFER END]")  # 로그 추가
                # clean up
                del feats, feats0, p_len
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return audio1
        except Exception as e:  # 예외 처리 블록
            print(f"[VC ERROR] Error during voice conversion: {e}")  # 오류 메시지 출력
            traceback.print_exc()  # 전체 traceback 정보 출력
            return np.zeros_like(audio0, dtype=np.float32)  # 오류 발생 시 0으로 채워진 NumPy 배열 반환 (오류 전파 방지)

    def _retrieve_speaker_embeddings(self, feats, index, big_npy, index_rate):
        print("[INDEX RETRIEVAL FUNCTION START]") # Log entry into the function
        try:
            npy = feats[0].cpu().numpy()
            faiss.omp_set_num_threads(1)
            print(f"[INDEX RETRIEVAL] feats[0] device after cpu(): {feats[0].device}") # Check device after .cpu()
            print("[INDEX RETRIEVAL] Index search starting...") # Log before index.search
            score, ix = index.search(npy, k=8)
            print("[INDEX RETRIEVAL] Index search finished.") # Log after index.search
            weight = np.square(1 / score)
            weight /= weight.sum(axis=1, keepdims=True)
            npy = np.sum(big_npy[ix] * np.expand_dims(weight, axis=2), axis=1)
            feats = (
                torch.from_numpy(npy).unsqueeze(0).to(self.device) * index_rate
                + (1 - index_rate) * feats
            )
            print(f"[INDEX RETRIEVAL FUNCTION END] Retrieved feats device: {feats.device}") # Log exit and feats device
            return feats
        except Exception as e:
            print(f"[INDEX RETRIEVAL ERROR] Error during speaker embedding retrieval: {e}") # Error log inside the function
            traceback.print_exc()
            return feats # Return original feats in case of error, to avoid pipeline crash

    def pipeline(
        self,
        model,
        net_g,
        sid,
        audio,
        pitch,
        f0_method,
        file_index,
        index_rate,
        pitch_guidance,
        volume_envelope,
        version,
        protect,
        hop_length,
        f0_autotune,
        f0_autotune_strength,
        f0_file,
    ):
        """The main pipeline function with try-except for error capture."""
        index = None
        big_npy = None
        if file_index != "" and os.path.exists(file_index) and index_rate > 0:
            try:
                index = faiss.read_index(file_index)
                big_npy = index.reconstruct_n(0, index.ntotal)
            except Exception as e:
                print(f"[PIPELINE INDEX ERROR] An error occurred reading the FAISS index: {e}")  # 오류 로그 강화
                traceback.print_exc()  # traceback 정보 출력
                index = big_npy = None
        else:
            index = big_npy = None

        audio = signal.filtfilt(bh, ah, audio)
        audio_pad = np.pad(audio, (self.window // 2, self.window // 2), mode="reflect")
        opt_ts = []
        if audio_pad.shape[0] > self.t_max:
            audio_sum = np.zeros_like(audio)
            for i in range(self.window):
                audio_sum += audio_pad[i : i - self.window]
            for t in range(self.t_center, audio.shape[0], self.t_center):
                opt_ts.append(
                    t
                    - self.t_query
                    + np.where(
                        np.abs(audio_sum[t - self.t_query : t + self.t_query])
                        == np.abs(audio_sum[t - self.t_query : t + self.t_query]).min()
                    )[0][0]
                )
        s = 0
        audio_opt = []
        t = None
        audio_pad = np.pad(audio, (self.t_pad, self.t_pad), mode="reflect")
        p_len = audio_pad.shape[0] // self.window
        inp_f0 = None
        if hasattr(f0_file, "name"):
            try:
                with open(f0_file.name, "r") as f:
                    lines = f.read().strip("\n").split("\n")
                inp_f0 = []
                for line in lines:
                    inp_f0.append([float(i) for i in line.split(",")])
                inp_f0 = np.array(inp_f0, dtype="float32")
            except Exception as e:
                print(f"[PIPELINE F0 FILE ERROR] An error occurred reading the F0 file: {e}")  # 오류 로그 강화
                traceback.print_exc()  # traceback 정보 출력
        sid = torch.tensor(sid, device=self.device).unsqueeze(0).long()
        print(f"Speaker ID 텐서 device: {sid.device}")

        pitch_coarse, pitch_original = None, None
        if pitch_guidance and f0_method is not None:
            pitch_coarse, pitch_original = self.get_f0(
                "input_audio_path",
                audio_pad,
                p_len,
                pitch,
                f0_method,
                hop_length,
                f0_autotune,
                f0_autotune_strength,
                inp_f0,
            )
            if pitch_coarse is not None and pitch_original is not None:
                print(
                    f"F0 coarse 텐서 device: {pitch_coarse.device if isinstance(pitch_coarse, torch.Tensor) else 'NumPy Array'}"
                )
                print(
                    f"F0 original 텐서 device: {pitch_original.device if isinstance(pitch_original, torch.Tensor) else 'NumPy Array'}"
                )
                pitch_coarse = pitch_coarse[:p_len]
                pitch_original = pitch_original[:p_len]
                if self.device == "mps":
                    pitch_original = torch.from_numpy(pitch_original).float()
                pitch_tensor = torch.tensor(pitch_coarse, device=self.device).unsqueeze(0).long()
                pitchf_tensor = torch.tensor(pitch_original, device=self.device).unsqueeze(0).float()
                print(f"F0 coarse 텐서 (최종) device: {pitch_tensor.device}")
                print(f"F0 original 텐서 (최종) device: {pitchf_tensor.device}")
            else:
                pitch_guidance = False
                pitch_tensor, pitchf_tensor = None, None
        else:
            pitch_tensor, pitchf_tensor = None, None

        try:  # pipeline 함수 전체를 try-except 로 감싸기
            for t in opt_ts:
                t = t // self.window * self.window
                if pitch_guidance and pitch_tensor is not None and pitchf_tensor is not None:
                    audio_opt.append(
                        self.voice_conversion(
                            model,
                            net_g,
                            sid,
                            audio_pad[s : t + self.t_pad2 + self.window],
                            pitch_tensor[:, s // self.window : (t + self.t_pad2) // self.window]
                            if pitch_tensor is not None
                            else None,
                            pitchf_tensor[:, s // self.window : (t + self.t_pad2) // self.window]
                            if pitchf_tensor is not None
                            else None,
                            index,
                            big_npy,
                            index_rate,
                            version,
                            protect,
                        )[self.t_pad_tgt : -self.t_pad_tgt]
                    )
                else:
                    audio_opt.append(
                        self.voice_conversion(
                            model,
                            net_g,
                            sid,
                            audio_pad[s : t + self.t_pad2 + self.window],
                            None,
                            None,
                            index,
                            big_npy,
                            index_rate,
                            version,
                            protect,
                        )[self.t_pad_tgt : -self.t_pad_tgt]
                    )
                s = t
            if pitch_guidance and pitch_tensor is not None and pitchf_tensor is not None:
                audio_opt.append(
                    self.voice_conversion(
                        model,
                        net_g,
                        sid,
                        audio_pad[t:],
                        pitch_tensor[:, t // self.window :] if t is not None else pitch_tensor,
                        pitchf_tensor[:, t // self.window :] if t is not None else pitchf_tensor,
                        index,
                        big_npy,
                        index_rate,
                        version,
                        protect,
                    )[self.t_pad_tgt : -self.t_pad_tgt]
                )
            else:
                audio_opt.append(
                    self.voice_conversion(
                        model,
                        net_g,
                        sid,
                        audio_pad[t:],
                        None,
                        None,
                        index,
                        big_npy,
                        index_rate,
                        version,
                        protect,
                        )[self.t_pad_tgt : -self.t_pad_tgt]
                )
            audio_opt = np.concatenate(audio_opt)
            if volume_envelope != 1:
                audio_opt = AudioProcessor.change_rms(
                    audio, self.sample_rate, audio_opt, self.sample_rate, volume_envelope
                )
            audio_max = np.abs(audio_opt).max() / 0.99
            if audio_max > 1:
                audio_opt /= audio_max
            if pitch_guidance and pitch_tensor is not None and pitchf_tensor is not None:
                del pitch_tensor, pitchf_tensor
            del sid
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return audio_opt

        except Exception as e:  # pipeline 함수 전체 예외 처리
            print(f"[PIPELINE ERROR] Error in pipeline function: {e}")  # 오류 로그 강화
            traceback.print_exc()  # traceback 정보 출력
            return np.zeros_like(audio, dtype=np.float32)