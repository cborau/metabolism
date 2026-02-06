from __future__ import annotations

import pickle
from pathlib import Path


class DummyModelParameterConfig:
    pass


class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "helper_module" and name == "ModelParameterConfig":
            return DummyModelParameterConfig
        return super().find_class(module, name)


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    pickle_path = base_dir / "result_files" / "output_data_0.pickle"
    if not pickle_path.exists():
        raise FileNotFoundError(f"Pickle file not found: {pickle_path}")

    with pickle_path.open("rb") as file:
        data = SafeUnpickler(file).load()

    print(f"Loaded: {pickle_path}")
    print(f"Top-level type: {type(data)}")
    if hasattr(data, "keys"):
        keys = sorted(data.keys())
        print(f"Top-level keys ({len(keys)}):")
        for key in keys:
            value = data[key]
            shape = getattr(value, "shape", None)
            if shape is not None:
                print(f"  - {key}: shape={shape}")
            else:
                print(f"  - {key}: type={type(value)}")
    else:
        print(data)


if __name__ == "__main__":
    main()
