import abc
import json
import numpy as np
import torch
import hashlib
import logging
from typing import NamedTuple
from transformers import AutoTokenizer, AutoModel

logger = logging.getLogger(__name__)


class BaseEmbedder(abc.ABC):
    @abc.abstractmethod
    def embed(self, sequences: list[str]) -> np.ndarray:
        raise NotImplementedError

    @abc.abstractmethod
    def embedding_dim(self) -> int:
        raise NotImplementedError

    @abc.abstractmethod
    def model_version(self) -> str:
        raise NotImplementedError


class ESM2Embedder(BaseEmbedder):

    def __init__(
        self,
        model_tag: str = "facebook/esm2_t33_650M_UR50D",
        device: str | None = None,
        debug: bool = False,
        compile_model: bool = True,
    ):
        import threading

        self._lock = threading.Lock()
        self.debug = debug
        self.MODEL_TAG = model_tag

        if device is not None:
            if device not in ("cpu", "cuda"):
                raise ValueError(f"Invalid device: {device}")
            if device == "cuda" and not torch.cuda.is_available():
                raise ValueError("CUDA requested but not available")
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True

        logger.info("Device selected: %s", self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_TAG, use_fast=True)

        self.model = AutoModel.from_pretrained(
            self.MODEL_TAG,
            dtype=torch.float16,
            low_cpu_mem_usage=True,
        ).to(self.device)
        self.model.eval()

        if compile_model and self.device.type == "cuda":
            self.model = torch.compile(self.model)
            logger.info("Model compiled with torch.compile")

        self._model_version = self._compute_model_version()

        if self.debug:
            ok = self.verify_determinism("ACDE")
            logger.info(f"Determinism check passed: {ok}")

    def _compute_model_version(self) -> str:
        config_dict = self.model.config.to_dict()
        config_json = json.dumps(config_dict, sort_keys=True)
        config_hash = hashlib.sha256(config_json.encode()).hexdigest()
        return f"{self.MODEL_TAG}_{config_hash[:8]}"

    @property
    def model_version(self) -> str:
        return self._model_version

    @property
    def embedding_dim(self) -> int:
        return self.model.config.hidden_size

    def tokenize(self, sequences: list[str]) -> dict[str, torch.Tensor]:
        tokens = self.tokenizer(
            sequences, padding=True, truncation=True, return_tensors="pt"
        )
        return {
            k: v.pin_memory() if self.device.type == "cuda" else v
            for k, v in tokens.items()
        }

    def embed_tokenized(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> np.ndarray:
        with torch.no_grad():
            with self._lock:
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        pooled = self._mean_pool(outputs.last_hidden_state, input_ids, attention_mask)
        normalized = pooled / pooled.norm(p=2, dim=1, keepdim=True)
        return normalized.float().cpu().numpy()

    def embed(self, sequences: list[str]) -> np.ndarray:
        tokens = self.tokenize(sequences)
        input_ids = tokens["input_ids"].to(self.device, non_blocking=True)
        attention_mask = tokens["attention_mask"].to(self.device, non_blocking=True)
        return self.embed_tokenized(input_ids, attention_mask)

    def _mean_pool(self, hidden_states, input_ids, attention_mask):
        cls_id = self.tokenizer.cls_token_id
        eos_id = self.tokenizer.eos_token_id

        residue_mask = (
            attention_mask.bool() & (input_ids != cls_id) & (input_ids != eos_id)
        )
        mask = residue_mask.unsqueeze(-1).to(hidden_states.dtype)

        summed = (hidden_states * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1)
        pooled = summed / counts

        zero_mask = pooled.abs().sum(dim=1) == 0
        if zero_mask.any():
            logger.warning(
                f"{zero_mask.sum().item()} sequences produced zero embeddings"
            )

        return pooled

    def verify_determinism(self, sequence: str) -> bool:
        result_a = self.embed([sequence])
        result_b = self.embed([sequence])
        return np.allclose(result_a, result_b, atol=1e-6)


class AAProps(NamedTuple):
    mol_weight: float
    hydrophobicity: float
    charge: float
    polarity: float


class PhysiochemicalEmbedder(BaseEmbedder):
    AA_PROPS: dict[str, AAProps] = {
        "A": AAProps(71.08, 1.8, 0.0, 9.1),
        "C": AAProps(103.14, 2.5, 0.0, 5.5),
        "D": AAProps(115.09, -3.5, -1.0, 13.0),
        "E": AAProps(129.11, -3.5, -1.0, 12.3),
        "F": AAProps(147.17, 2.8, 0.0, 5.2),
        "G": AAProps(57.05, -0.4, 0.0, 9.0),
        "H": AAProps(137.14, -3.2, 0.5, 10.4),
        "I": AAProps(113.16, 4.5, 0.0, 5.2),
        "K": AAProps(128.17, -3.9, 1.0, 11.3),
        "L": AAProps(113.16, 3.8, 0.0, 4.9),
        "M": AAProps(131.20, 1.9, 0.0, 5.7),
        "N": AAProps(114.10, -3.5, 0.0, 11.6),
        "P": AAProps(97.12, -1.6, 0.0, 8.0),
        "Q": AAProps(128.13, -3.5, 0.0, 10.5),
        "R": AAProps(156.19, -4.5, 1.0, 10.5),
        "S": AAProps(87.08, -0.8, 0.0, 9.2),
        "T": AAProps(101.10, -0.7, 0.0, 8.6),
        "V": AAProps(99.13, 4.2, 0.0, 5.9),
        "W": AAProps(186.21, -0.9, 0.0, 5.4),
        "Y": AAProps(163.17, -1.3, 0.0, 6.2),
    }

    def __init__(self):
        arr = np.array(list(self.AA_PROPS.values()), dtype=np.float32)
        self._unknown_props = arr.mean(axis=0)

    @property
    def embedding_dim(self) -> int:
        return 4

    @property
    def model_version(self) -> str:
        return "physiochemical_v1"

    def embed(self, sequences: list[str]) -> np.ndarray:
        results = []

        for seq in sequences:
            vecs = []

            for aa in seq:
                props = self.AA_PROPS.get(aa, self._unknown_props)
                vecs.append(props)

            if len(vecs) == 0:
                results.append(self._unknown_props)
                continue

            mat = np.array(vecs, dtype=np.float32)

            vec = mat.mean(axis=0)
            results.append(vec)

        # (N, 4)
        X = np.stack(results)

        # (N, 1)
        norm = np.linalg.norm(X, axis=1, keepdims=True)

        eps = 1e-8

        # (N, )
        zero_mask = (norm < eps).reshape(-1)

        if np.any(zero_mask):
            logger.warning("%d zero-norm embedding detected", int(zero_mask.sum()))

        X = X / np.maximum(norm, eps)

        return X


class RandomEmbedder(BaseEmbedder):

    def __init__(self, seed: int = 42) -> None:
        self._seed = seed
        self._rng = np.random.default_rng(seed)

    @property
    def embedding_dim(self) -> int:
        return 1280

    @property
    def model_version(self) -> str:
        return f"random_seed{self._seed}"

    def embed(self, sequences: list[str]) -> np.ndarray:
        N = len(sequences)

        X = self._rng.standard_normal(size=(N, 1280)).astype(np.float32)

        norm = np.linalg.norm(X, axis=1, keepdims=True)
        norm = np.maximum(norm, 1e-8)

        return (X / norm).astype(np.float32)
