import numpy as np

from scripts.run_speech_commands_real_baselines import LABELS, Records, fit, metric


def main() -> None:
    labels = np.repeat(np.arange(len(LABELS)), 2)
    values = np.random.default_rng(68).integers(-2000, 2001, size=(len(labels), 3200), dtype=np.int16)
    records = Records(values, labels, np.arange(len(labels)))
    assert 0.0 <= metric(fit("logreg_0.1", records.values, records.labels), records)["macro_f1"] <= 1.0
    print("SUCCESS: Speech Commands observed baseline helper emits finite probabilities")


if __name__ == "__main__":
    main()
