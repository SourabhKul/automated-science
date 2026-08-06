import io
import struct

from scripts.run_emnist_source_gate import IMAGE_MAGIC, LABEL_MAGIC, read_image_header, read_label_header


def main() -> None:
    assert read_image_header(io.BytesIO(struct.pack(">IIII", IMAGE_MAGIC, 112_800, 28, 28))) == (112_800, 28, 28)
    assert read_label_header(io.BytesIO(struct.pack(">II", LABEL_MAGIC, 112_800))) == 112_800
    print("SUCCESS: EMNIST IDX parsers enforce the fixed raw-grid header contract")


if __name__ == "__main__":
    main()
