from pathlib import Path

import yaml


RESOURCE_TYPE_MAPPING: dict[str, str] = {
	"survey": "resources",
	"surveys": "resources",
	"library": "resources",
	"libraries": "resources",
	"resource": "resources",
	"resources": "resources",
	"tutorial": "resources",
	"tutorials": "resources",
	"course": "resources",
	"courses": "resources",
	"corpus": "resources",
	"corpora": "resources",
	"lecture": "resources",
	"lectures": "resources",
	"paper": "resources",
	"papers": "resources",
	"other": "resources",
	"book": "resources",
	"books": "resources",
	"naclo": "assignment",
	"naclo problems": "assignment",
}


def find_project_root(start_path: Path | None = None, marker: str = "config.yaml") -> Path:
	current = (start_path or Path(__file__).resolve()).parent
	for candidate in [current, *current.parents]:
		if (candidate / marker).exists():
			return candidate
	raise FileNotFoundError(f"Could not find {marker!r} starting from {current}")


def load_config(config_path: str | Path | None = None) -> tuple[dict, Path]:
	"""Load YAML config and return (config_dict, project_root)."""
	if config_path is None:
		project_root = find_project_root()
		config_file = project_root / "config.yaml"
	else:
		config_file = Path(config_path).expanduser().resolve()
		project_root = config_file.parent

	with config_file.open("r", encoding="utf-8") as file:
		return yaml.safe_load(file) or {}, project_root


def get_config_value(config: dict, *keys: str, default=None):
	"""Get a nested value from config using keys like ('data_paths', 'path')."""
	value = config
	for key in keys:
		if not isinstance(value, dict) or key not in value:
			return default
		value = value[key]
	return value


def map_resource_medium_to_type(medium: str | None, default: str | None = None) -> str | None:
	"""Normalize a resource medium into the coarse content types documented in README."""
	if medium is None:
		return default

	normalized_medium = str(medium).strip().lower()
	if not normalized_medium:
		return default

	return RESOURCE_TYPE_MAPPING.get(normalized_medium, default)

if __name__ == "__main__":
	cfg, root = load_config()
	print("Project Root:", root)
	print("Full Config Test:", cfg)

	print("Embeddings :", get_config_value(cfg,"embeddings" ))
	print("Data Paths:", get_config_value(cfg, "data_paths", "path"))

