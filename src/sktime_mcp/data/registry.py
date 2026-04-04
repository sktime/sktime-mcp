"""
Registry for data source adapters.

Manages registration and creation of data source adapters.
"""

from .adapters import FileAdapter, PandasAdapter, SQLAdapter, UrlAdapter
from .base import DataSourceAdapter


class DataSourceRegistry:
    """
    Registry for data source adapters.

    Provides a central place to register and retrieve adapters.
    """

    _adapters: dict[str, type[DataSourceAdapter]] = {
        "pandas": PandasAdapter,
        "sql": SQLAdapter,
        "file": FileAdapter,
        "url": UrlAdapter,
    }

    @classmethod
    def register(cls, name: str, adapter_class: type[DataSourceAdapter]):
        """
        Register a new adapter.

        Args:
            name: Name to register the adapter under
            adapter_class: Adapter class (must inherit from DataSourceAdapter)
        """
        if not issubclass(adapter_class, DataSourceAdapter):
            raise TypeError(
                f"Adapter class must inherit from DataSourceAdapter, got {adapter_class}"
            )

        cls._adapters[name] = adapter_class

    @classmethod
    def get_adapter(cls, source_type: str) -> type[DataSourceAdapter]:
        """
        Get adapter class by type.

        Args:
            source_type: Type of data source (e.g., "pandas", "sql", "file")

        Returns:
            Adapter class

        Raises:
            ValueError: If source type is not registered
        """
        if source_type not in cls._adapters:
            available = ", ".join(cls._adapters.keys())
            raise ValueError(
                f"Unknown data source type: '{source_type}'. Available types: {available}"
            )

        return cls._adapters[source_type]

    @classmethod
    def create_adapter(cls, config: dict) -> DataSourceAdapter:
        """
        Create adapter instance from config.

        Args:
            config: Configuration dictionary with 'type' key

        Returns:
            Adapter instance

        Raises:
            ValueError: If config is invalid or type is not registered
        """
        if not isinstance(config, dict):
            raise ValueError(f"Config must be a dictionary, got {type(config)}")

        source_type = config.get("type")
        if not source_type:
            raise ValueError("Config must specify 'type' key")

        adapter_class = cls.get_adapter(source_type)
        return adapter_class(config)

    @classmethod
    def list_adapters(cls) -> list:
        """
        List all registered adapter types.

        Returns:
            List of adapter type names
        """
        return list(cls._adapters.keys())

    @classmethod
    def get_adapter_info(cls, source_type: str) -> dict:
        """
        Get information about an adapter.

        Args:
            source_type: Type of data source

        Returns:
            Dictionary with adapter information
        """
        adapter_class = cls.get_adapter(source_type)

        return {
            "type": source_type,
            "class": adapter_class.__name__,
            "module": adapter_class.__module__,
            "docstring": adapter_class.__doc__,
        }
