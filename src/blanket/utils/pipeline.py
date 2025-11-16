import logging
from pathlib import Path

from datasets import Dataset
from hydra.experimental.callback import Callback
from jinja2 import Environment, PackageLoader
from omegaconf import DictConfig, OmegaConf

from blanket.utils import read_json


class FinalCallback(Callback):
    """Hydra callback to log when multirun is finished and create HF dataset README."""

    def __init__(self) -> None:
        self.log = logging.getLogger(__name__)

    def on_multirun_start(self, config: DictConfig, **kwargs) -> None:
        """Handle overwrite flag at the start of multirun."""
        import shutil

        if config.get("overwrite", False):
            output_path = Path(config.output_path)
            if output_path.exists():
                self.log.info(
                    "Overwrite flag set. Removing existing output directory: %s",
                    output_path,
                )
                shutil.rmtree(output_path)
                self.log.info("✓ Output directory removed")

    def _get_data_dirs(self, output_path: Path) -> list[Path]:
        """Get all data directories sorted by name."""
        return sorted(
            [
                p
                for p in output_path.iterdir()
                if p.is_dir() and p.name.startswith("data_")
            ]
        )

    def _create_yaml_header(self, data_dirs: list[Path]) -> DictConfig:
        """Create YAML header configuration for HuggingFace dataset."""
        yaml_header = {
            "configs": [
                {"config_name": data_dir.name, "data_dir": data_dir.name}
                for data_dir in data_dirs
            ]
        }
        return OmegaConf.create(yaml_header)

    def _create_metadata_parquet(
        self, output_path: Path, data_dirs: list[Path]
    ) -> None:
        """Create merged metadata parquet file from all data directories."""
        metadata_records = []

        for data_dir in data_dirs:
            meta_path = data_dir / "meta.json"
            if meta_path.exists():
                metadata = read_json(meta_path)
                metadata_records.append(metadata)
            else:
                self.log.warning("Metadata file not found: %s", meta_path)

        # Create HuggingFace Dataset from metadata records
        if metadata_records:
            dataset = Dataset.from_dict(
                {
                    k: [record.get(k) for record in metadata_records]
                    for k in metadata_records[0]
                }
            )
            metadata_path = str(output_path / "metadata.parquet")
            dataset.to_parquet(metadata_path)
            self.log.info("Metadata saved to %s", metadata_path)
        else:
            self.log.warning("No metadata records found")

    def _write_readme(
        self,
        readme_path: Path,
        yaml_header: DictConfig,
        data_dirs: list[Path],
        config: DictConfig,
    ) -> None:
        """Write README.md with YAML header and markdown content using Jinja2 template."""
        # Setup Jinja2 environment
        env = Environment(
            loader=PackageLoader("blanket.utils", "templates"),
            keep_trailing_newline=True,
        )
        template = env.get_template("README.md.j2")

        # Prepare template context
        context = {
            "yaml_header": OmegaConf.to_yaml(yaml_header),
            "dataset_count": len(data_dirs),
            "config_yaml": OmegaConf.to_yaml(config),
            "dataset_name": config["dataset_name"],
        }

        # Render and write template
        readme_content = template.render(**context)
        with open(readme_path, "w") as f:
            f.write(readme_content)

    def _clean_config(self, config: DictConfig) -> DictConfig:
        """Remove Hydra-specific entries from configuration and merge sweeper params."""
        cfg_dict = OmegaConf.to_container(config)

        # Extract sweeper params before removing hydra
        sweeper_params = cfg_dict.get("hydra", {}).get("sweeper", {}).get("params", {})

        # Remove hydra entirely
        cfg_dict.pop("hydra", None)

        # Merge sweeper params into main config (replace existing or add new)
        for param_name, param_value in sweeper_params.items():
            cfg_dict[param_name] = param_value

        return OmegaConf.create(cfg_dict)

    def on_multirun_end(self, config: DictConfig, **kwargs) -> None:
        # Save configuration to output directory for reproducibility
        output_path = Path(config.output_path)
        # config_path = output_path / "config.yaml"
        cfg_to_save = self._clean_config(config)
        # OmegaConf.save(cfg_to_save, config_path)
        # self.log.info("Configuration saved to %s", config_path)

        # Get data directories and create README
        data_dirs = self._get_data_dirs(output_path)
        yaml_header = self._create_yaml_header(data_dirs)

        readme_path = output_path / "README.md"
        self._write_readme(readme_path, yaml_header, data_dirs, cfg_to_save)

        # Create merged metadata parquet
        self._create_metadata_parquet(output_path, data_dirs)

        self.log.info("README.md created at %s", readme_path)
        self.log.info(
            "Done! Generated %d Datasets in %s",
            len(data_dirs),
            output_path,
        )
