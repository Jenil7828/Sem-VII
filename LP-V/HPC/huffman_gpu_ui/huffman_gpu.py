from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import base64
import heapq
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from numba import cuda
except Exception:  # pragma: no cover - numba may not be installed yet
    cuda = None


@dataclass
class CompressionResult:
    original_text: str
    original_bytes: bytes
    encoded_bytes: bytes
    total_encoded_bits: int
    compression_ratio: float
    codebook: Dict[int, str]
    used_gpu: bool


@dataclass
class _Node:
    freq: int
    symbol: Optional[int] = None
    left: Optional["_Node"] = None
    right: Optional["_Node"] = None


def _build_huffman_tree(data: bytes) -> Optional[_Node]:
    if not data:
        return None

    frequency = Counter(data)
    heap: List[Tuple[int, int, _Node]] = []
    tie_breaker = 0

    for symbol, freq in frequency.items():
        heapq.heappush(heap, (freq, tie_breaker, _Node(freq=freq, symbol=symbol)))
        tie_breaker += 1

    if len(heap) == 1:
        only = heap[0][2]
        return _Node(freq=only.freq, left=only)

    while len(heap) > 1:
        f1, _, n1 = heapq.heappop(heap)
        f2, _, n2 = heapq.heappop(heap)
        merged = _Node(freq=f1 + f2, left=n1, right=n2)
        heapq.heappush(heap, (merged.freq, tie_breaker, merged))
        tie_breaker += 1

    return heap[0][2]


def _generate_codes(root: Optional[_Node]) -> Dict[int, str]:
    if root is None:
        return {}

    codes: Dict[int, str] = {}

    def walk(node: _Node, prefix: str) -> None:
        if node.symbol is not None:
            codes[node.symbol] = prefix if prefix else "0"
            return
        if node.left is not None:
            walk(node.left, prefix + "0")
        if node.right is not None:
            walk(node.right, prefix + "1")

    walk(root, "")
    return codes


def _codebook_to_lookup(codebook: Dict[int, str]) -> Tuple[np.ndarray, np.ndarray]:
    bits_lookup = np.zeros(256, dtype=np.uint64)
    len_lookup = np.zeros(256, dtype=np.uint8)

    for symbol, bitstring in codebook.items():
        value = 0
        for bit in bitstring:
            value = (value << 1) | (1 if bit == "1" else 0)
        bits_lookup[symbol] = np.uint64(value)
        len_lookup[symbol] = np.uint8(len(bitstring))

    return bits_lookup, len_lookup


if cuda is not None:

    @cuda.jit
    def _encode_kernel(symbols, bits_lut, lens_lut, out_bits, out_lens):
        i = cuda.grid(1)
        if i < symbols.size:
            sym = symbols[i]
            out_bits[i] = bits_lut[sym]
            out_lens[i] = lens_lut[sym]


def _encode_symbols_gpu_or_cpu(data: bytes, codebook: Dict[int, str]) -> Tuple[np.ndarray, np.ndarray, bool]:
    symbols = np.frombuffer(data, dtype=np.uint8)
    bits_lut, lens_lut = _codebook_to_lookup(codebook)

    if symbols.size == 0:
        return np.array([], dtype=np.uint64), np.array([], dtype=np.uint8), False

    if cuda is not None and cuda.is_available():
        d_symbols = cuda.to_device(symbols)
        d_bits_lut = cuda.to_device(bits_lut)
        d_lens_lut = cuda.to_device(lens_lut)

        d_out_bits = cuda.device_array(symbols.size, dtype=np.uint64)
        d_out_lens = cuda.device_array(symbols.size, dtype=np.uint8)

        threads = 256
        blocks = (symbols.size + threads - 1) // threads
        _encode_kernel[blocks, threads](d_symbols, d_bits_lut, d_lens_lut, d_out_bits, d_out_lens)

        out_bits = d_out_bits.copy_to_host()
        out_lens = d_out_lens.copy_to_host()
        return out_bits, out_lens, True

    out_bits = bits_lut[symbols]
    out_lens = lens_lut[symbols]
    return out_bits.astype(np.uint64), out_lens.astype(np.uint8), False


def _pack_codes(codes: np.ndarray, lengths: np.ndarray) -> Tuple[bytes, int]:
    out = bytearray()
    bit_buffer = 0
    bit_count = 0

    for code, length in zip(codes, lengths):
        l = int(length)
        if l == 0:
            continue
        bit_buffer = (bit_buffer << l) | int(code)
        bit_count += l

        while bit_count >= 8:
            shift = bit_count - 8
            out.append((bit_buffer >> shift) & 0xFF)
            bit_buffer &= (1 << shift) - 1
            bit_count -= 8

    if bit_count > 0:
        out.append((bit_buffer << (8 - bit_count)) & 0xFF)

    total_bits = int(np.sum(lengths, dtype=np.uint64))
    return bytes(out), total_bits


def _to_bitstring(encoded_bytes: bytes, total_bits: int) -> str:
    if not encoded_bytes or total_bits == 0:
        return ""
    joined = "".join(f"{b:08b}" for b in encoded_bytes)
    return joined[:total_bits]


def compress_text(text: str) -> CompressionResult:
    source = text.encode("utf-8")
    if not source:
        return CompressionResult(
            original_text=text,
            original_bytes=b"",
            encoded_bytes=b"",
            total_encoded_bits=0,
            compression_ratio=0.0,
            codebook={},
            used_gpu=False,
        )

    tree = _build_huffman_tree(source)
    codebook = _generate_codes(tree)

    codes, lengths, used_gpu = _encode_symbols_gpu_or_cpu(source, codebook)
    encoded_bytes, total_bits = _pack_codes(codes, lengths)

    original_size = len(source)
    compressed_size = len(encoded_bytes)
    ratio = (compressed_size / original_size) if original_size else 0.0

    return CompressionResult(
        original_text=text,
        original_bytes=source,
        encoded_bytes=encoded_bytes,
        total_encoded_bits=total_bits,
        compression_ratio=ratio,
        codebook=codebook,
        used_gpu=used_gpu,
    )


def decompress_text(result: CompressionResult) -> str:
    if not result.original_bytes:
        return ""

    reverse = {code: symbol for symbol, code in result.codebook.items()}
    bits = _to_bitstring(result.encoded_bytes, result.total_encoded_bits)

    decoded = bytearray()
    cursor = ""
    for bit in bits:
        cursor += bit
        symbol = reverse.get(cursor)
        if symbol is not None:
            decoded.append(symbol)
            cursor = ""

    return decoded.decode("utf-8", errors="replace")


def codebook_rows(result: CompressionResult) -> List[Dict[str, str]]:
    if not result.codebook:
        return []

    frequency = Counter(result.original_bytes)
    rows = []

    for symbol, code in sorted(result.codebook.items(), key=lambda item: (len(item[1]), item[1])):
        display = chr(symbol) if 32 <= symbol <= 126 else f"0x{symbol:02X}"
        rows.append(
            {
                "symbol": display,
                "byte": str(symbol),
                "frequency": str(frequency[symbol]),
                "code": code,
                "length": str(len(code)),
            }
        )

    return rows


def build_export_payload(result: CompressionResult) -> bytes:
    lines = [
        "Huffman GPU Mini Project Export",
        f"backend={'GPU' if result.used_gpu else 'CPU'}",
        f"original_size={len(result.original_bytes)}",
        f"compressed_size={len(result.encoded_bytes)}",
        f"total_bits={result.total_encoded_bits}",
        "codebook:",
    ]

    for symbol, code in sorted(result.codebook.items()):
        lines.append(f"{symbol}:{code}")

    lines.append("payload_base64:")
    lines.append(base64.b64encode(result.encoded_bytes).decode("ascii"))

    return "\n".join(lines).encode("utf-8")
