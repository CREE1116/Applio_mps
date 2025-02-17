
import torch
import json
import os

version_config_paths = [
    os.path.join("48000.json"),
    os.path.join("40000.json"),
    os.path.join("44100.json"),
    os.path.join("32000.json"),
]


def singleton(cls):
    instances = {}

    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance


@singleton
class Config:
    def __init__(self):
        # 사용 가능한 디바이스 설정 (CUDA, MPS, CPU 순서로 확인)
        if torch.cuda.is_available():
            self.device = "cuda:0"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        # GPU 이름 설정 (CUDA 또는 MPS 사용 시)
        if self.device.startswith("cuda"):
            self.gpu_name = torch.cuda.get_device_name(int(self.device.split(":")[-1]))
        elif self.device == "mps":
            self.gpu_name = "Apple M-series GPU"  # MPS GPU 이름
        else:
            self.gpu_name = None

        self.json_config = self.load_config_json()
        self.gpu_mem = None
        self.x_pad, self.x_query, self.x_center, self.x_max = self.device_config()

        # 어떤 디바이스를 사용하는지 출력
        print(f"디바이스 설정 완료: {self.device}")
        if self.gpu_name:
            print(f"GPU 이름: {self.gpu_name}")
        else:
            print("GPU를 사용하지 않습니다.")


    def load_config_json(self):
        configs = {}
        for config_file in version_config_paths:
            config_path = os.path.join("rvc", "configs", config_file)
            with open(config_path, "r") as f:
                configs[config_file] = json.load(f)
        return configs

    def device_config(self):
        if self.device.startswith("cuda"):
            self.set_cuda_config()
        elif self.device == "mps":
            self.set_mps_config()  # MPS 설정 메소드 호출
        else:
            self.device = "cpu"

        # 6GB GPU 메모리 기준 설정 (CUDA 기준일 가능성이 높음)
        x_pad, x_query, x_center, x_max = (1, 6, 38, 41)
        if self.gpu_mem is not None and self.gpu_mem <= 4: # CUDA 메모리 체크 기준
            # 5GB GPU 메모리 기준 설정
            x_pad, x_query, x_center, x_max = (1, 5, 30, 32)

        return x_pad, x_query, x_center, x_max

    def set_cuda_config(self):
        i_device = int(self.device.split(":")[-1])
        self.gpu_name = torch.cuda.get_device_name(i_device)
        self.gpu_mem = torch.cuda.get_device_properties(i_device).total_memory // (
            1024**3
        )

    def set_mps_config(self):
        # MPS는 CUDA처럼 직접적인 메모리 정보를 얻기 어려울 수 있습니다.
        # 일단 gpu_mem은 None으로 두고, 필요하다면 시스템 메모리 정보를 가져오도록 수정할 수 있습니다.
        # 여기서는 MPS 사용 중임을 나타내기 위해 gpu_name만 설정합니다.
        self.gpu_name = "Apple M-series GPU (MPS 백엔드 사용)"
        self.gpu_mem = None # MPS 메모리 정보는 일단 None으로 설정


def max_vram_gpu(gpu):
    if torch.cuda.is_available():
        gpu_properties = torch.cuda.get_device_properties(gpu)
        total_memory_gb = round(gpu_properties.total_memory / 1024 / 1024 / 1024)
        return total_memory_gb
    else:
        return "8" # CUDA를 사용하지 않을 때 기본값, MPS 환경에 따라 조정 필요할 수 있음


def get_gpu_info():
    ngpu = torch.cuda.device_count()
    gpu_infos = []
    if torch.cuda.is_available() or ngpu != 0:
        for i in range(ngpu):
            gpu_name = torch.cuda.get_device_name(i)
            mem = int(
                torch.cuda.get_device_properties(i).total_memory / 1024 / 1024 / 1024
                + 0.4
            )
            gpu_infos.append(f"{i}: {gpu_name} ({mem} GB)")
    elif torch.backends.mps.is_available():
        gpu_infos.append(f"MPS 장치: Apple M-series GPU (MPS 백엔드 사용)") # MPS 사용 정보 추가
    if len(gpu_infos) > 0:
        gpu_info = "\n".join(gpu_infos)
    else:
        gpu_info = "죄송합니다, 학습을 지원하는 호환 가능한 GPU를 찾을 수 없습니다."
    return gpu_info


def get_number_of_gpus():
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        return "-".join(map(str, range(num_gpus)))
    elif torch.backends.mps.is_available():
        return "mps" # MPS 사용 시 "mps" 반환, 필요에 따라 "1-mps" 등으로 변경 가능
    else:
        return "-"