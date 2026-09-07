from __future__ import annotations

from typing import Dict, Any, List, Optional
import json
import os
import base64
import hashlib
import logging

from ..db import get_cursor

logger = logging.getLogger(__name__)


class SettingsRepository:
    """Key-value settings storage with enhanced dual-mode research agent configuration support."""

    @staticmethod
    def get(key: str) -> str | None:
        with get_cursor() as cur:
            cur.execute("SELECT value FROM settings WHERE key = %s", (key,))
            row = cur.fetchone()
            return row["value"] if row else None

    @staticmethod
    def set(key: str, value: str) -> None:
        with get_cursor() as cur:
            cur.execute(
                """
                INSERT INTO settings (key, value) VALUES (%s,%s)
                ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value
                """,
                (key, value),
            )

    @staticmethod
    def get_int(key: str, default: int = 0) -> int:
        """Get a setting value as integer with fallback."""
        value = SettingsRepository.get(key)
        if value is None:
            return default
        try:
            return int(value)
        except (ValueError, TypeError):
            return default

    @staticmethod
    def get_float(key: str, default: float = 0.0) -> float:
        """Get a setting value as float with fallback."""
        value = SettingsRepository.get(key)
        if value is None:
            return default
        try:
            return float(value)
        except (ValueError, TypeError):
            return default

    @staticmethod
    def get_bool(key: str, default: bool = False) -> bool:
        """Get a setting value as boolean with fallback."""
        value = SettingsRepository.get(key)
        if value is None:
            return default
        return value.lower() in ('true', '1', 'yes', 'on')

    @staticmethod
    def delete(key: str) -> None:
        with get_cursor() as cur:
            cur.execute("DELETE FROM settings WHERE key = %s", (key,))

    @staticmethod
    def all() -> Dict[str, str]:
        with get_cursor() as cur:
            cur.execute("SELECT key, value FROM settings")
            rows = cur.fetchall()
            return {row["key"]: row["value"] for row in rows}

    @staticmethod
    def _encrypt(plaintext: str) -> str:
        from ..security import encrypt
        return encrypt(plaintext)

    @staticmethod
    def _decrypt(ciphertext: str) -> str:
        from ..security import decrypt
        return decrypt(ciphertext)

    @staticmethod
    def set_secret_setting(key: str, value: str) -> None:
        SettingsRepository.set(key, SettingsRepository._encrypt(value))

    @staticmethod
    def get_secret_setting(key: str) -> str | None:
        value = SettingsRepository.get(key)
        if not value:
            return None
        from ..security import PREFIX
        if not value.startswith(PREFIX) and os.getenv(key):
            # Existing environment credentials remain usable until the operator
            # explicitly selects the untagged legacy storage format to migrate.
            return os.environ[key]
        return SettingsRepository._decrypt(value)

    # Convenience helpers mirroring old API

    @staticmethod
    def get_email_recipients() -> List[str]:
        val = SettingsRepository.get('email_recipients')
        return json.loads(val) if val else []

    @staticmethod
    def set_email_recipients(recipients: List[str]):
        SettingsRepository.set('email_recipients', json.dumps(recipients))

    @staticmethod
    def get_visualizer_settings() -> Dict[str, Any]:
        val = SettingsRepository.get('visualizer_settings')
        return json.loads(val) if val else {}

    @staticmethod
    def set_visualizer_settings(settings: Dict[str, Any]):
        SettingsRepository.set('visualizer_settings', json.dumps(settings))

    @staticmethod
    def get_performance_config() -> Dict[str, Any]:
        val = SettingsRepository.get('performance_config')
        if not val:
            return {}
        try:
            return json.loads(val)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse performance config: {e}")
            return {}

    # Enhanced dual-mode research agent configuration management
    
    @staticmethod
    def get_orchestration_config() -> Dict[str, Any]:
        """
        Get the complete orchestration configuration including dual-mode research agent settings.
        
        Returns:
            Orchestration configuration dictionary, or empty dict if not found
        """
        config_json = SettingsRepository.get("orchestration")
        if config_json:
            try:
                return json.loads(config_json)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse orchestration config: {e}")
                return {}
        return {}
    
    @staticmethod
    def set_orchestration_config(config: Dict[str, Any]) -> None:
        """
        Set the complete orchestration configuration.
        
        Args:
            config: Orchestration configuration dictionary
        """
        SettingsRepository.set("orchestration", json.dumps(config))
    
    @staticmethod
    def get_research_agent_mode() -> str:
        """
        Get the current research agent mode.
        
        Returns:
            "single" or "multi" based on configuration, defaults to "single"
        """
        config = SettingsRepository.get_orchestration_config()
        
        # Check for explicit mode setting
        mode = config.get("research_agent_mode")
        if mode in ["single", "multi"]:
            return mode
        
        # Infer mode from configuration presence
        if "multi_agent_config" in config:
            return "multi"
        elif "research_agent_model_config" in config:
            return "single"
        else:
            return "single"  # Default
    
    @staticmethod
    def set_research_agent_mode(mode: str) -> None:
        """
        Set the research agent mode.
        
        Args:
            mode: "single" or "multi"
        """
        if mode not in ["single", "multi"]:
            raise ValueError(f"Invalid research agent mode: {mode}. Must be 'single' or 'multi'")
        
        config = SettingsRepository.get_orchestration_config()
        config["research_agent_mode"] = mode
        SettingsRepository.set_orchestration_config(config)
    
    @staticmethod
    def get_single_agent_config() -> Dict[str, Any]:
        """
        Get single-agent research configuration.
        
        Returns:
            Single-agent configuration dictionary
        """
        config = SettingsRepository.get_orchestration_config()
        return config.get("single_agent_config", {})
    
    @staticmethod
    def set_single_agent_config(single_config: Dict[str, Any]) -> None:
        """
        Set single-agent research configuration.
        
        Args:
            single_config: Single-agent configuration dictionary
        """
        config = SettingsRepository.get_orchestration_config()
        config["single_agent_config"] = single_config
        
        # Also update the legacy research_agent_model_config for backward compatibility
        if "model_config" in single_config:
            config["research_agent_model_config"] = single_config["model_config"]
        
        SettingsRepository.set_orchestration_config(config)
    
    @staticmethod
    def get_multi_agent_config() -> Dict[str, Any]:
        """
        Get multi-agent research configuration.
        
        Returns:
            Multi-agent configuration dictionary
        """
        config = SettingsRepository.get_orchestration_config()
        return config.get("multi_agent_config", {})
    
    @staticmethod
    def set_multi_agent_config(multi_config: Dict[str, Any]) -> None:
        """
        Set multi-agent research configuration.
        
        Args:
            multi_config: Multi-agent configuration dictionary
        """
        config = SettingsRepository.get_orchestration_config()
        config["multi_agent_config"] = multi_config
        SettingsRepository.set_orchestration_config(config)
    
    @staticmethod
    def validate_research_agent_config() -> Dict[str, Any]:
        """
        Validate the current research agent configuration.
        
        Returns:
            Validation results dictionary with 'valid' boolean and 'issues' list
        """
        from ..research_agent.model_router import validate_dual_mode_config
        
        config = SettingsRepository.get_orchestration_config()
        return validate_dual_mode_config(config)
    
    @staticmethod
    def migrate_legacy_research_config() -> bool:
        """
        Migrate legacy research agent configuration to dual-mode format.
        
        Returns:
            True if migration was performed, False if no migration needed
        """
        config = SettingsRepository.get_orchestration_config()
        
        # Check if we already have dual-mode configuration
        if "research_agent_mode" in config:
            return False  # Already migrated
        
        # Check for legacy research_agent_model_config
        legacy_config = config.get("research_agent_model_config", {})
        if not legacy_config:
            return False  # No legacy config to migrate
        
        logger.info("Migrating legacy research agent configuration to dual-mode format")
        
        # Create dual-mode configuration
        config["research_agent_mode"] = "single"
        
        # Migrate to single_agent_config
        config["single_agent_config"] = {
            "model_config": legacy_config,
            "max_research_loops": 3,
            "max_research_context_tokens": 15000,
            "compress_to_ratio": 0.2,
            "search_config": {
                "local_limit": 20,
                "external_limit": 15
            }
        }
        
        # Create default multi_agent_config
        from ..research_agent.model_router import create_default_dual_mode_config
        default_config = create_default_dual_mode_config()
        config["multi_agent_config"] = default_config["multi_agent_config"]
        
        # Save migrated configuration
        SettingsRepository.set_orchestration_config(config)
        
        logger.info("Successfully migrated legacy research agent configuration")
        return True
    
    @staticmethod
    def ensure_dual_mode_config() -> None:
        """
        Ensure that dual-mode research agent configuration exists, creating defaults if needed.
        """
        config = SettingsRepository.get_orchestration_config()
        
        # First, try to migrate legacy configuration
        SettingsRepository.migrate_legacy_research_config()
        
        # Refresh config after potential migration
        config = SettingsRepository.get_orchestration_config()
        
        # Check if we need to create default configuration
        needs_defaults = False
        
        if "research_agent_mode" not in config:
            needs_defaults = True
        elif config.get("research_agent_mode") == "single" and "single_agent_config" not in config:
            needs_defaults = True
        elif config.get("research_agent_mode") == "multi" and "multi_agent_config" not in config:
            needs_defaults = True
        
        if needs_defaults:
            logger.info("Creating default dual-mode research agent configuration")
            
            from ..research_agent.model_router import create_default_dual_mode_config
            default_config = create_default_dual_mode_config()
            
            # Merge with existing config, preserving other settings
            for key, value in default_config.items():
                if key not in config:
                    config[key] = value
            
            SettingsRepository.set_orchestration_config(config)
            logger.info("Created default dual-mode research agent configuration")
    
    @staticmethod
    def get_effective_research_config() -> Dict[str, Any]:
        """
        Get the effective research configuration based on current mode.
        
        Returns:
            Configuration dictionary for the current research agent mode
        """
        config = SettingsRepository.get_orchestration_config()
        mode = SettingsRepository.get_research_agent_mode()
        
        if mode == "multi":
            return config.get("multi_agent_config", {})
        else:
            return config.get("single_agent_config", {})
    
    @staticmethod
    def switch_research_agent_mode(new_mode: str) -> Dict[str, Any]:
        """
        Switch research agent mode and return validation results.
        
        Args:
            new_mode: "single" or "multi"
            
        Returns:
            Validation results for the new mode
        """
        # Ensure dual-mode configuration exists
        SettingsRepository.ensure_dual_mode_config()
        
        # Set the new mode
        SettingsRepository.set_research_agent_mode(new_mode)
        
        # Validate the configuration
        validation_results = SettingsRepository.validate_research_agent_config()
        
        logger.info(f"Switched research agent mode to '{new_mode}', validation: {validation_results['valid']}")
        
        return validation_results
    
    @staticmethod
    def get_dual_mode_status() -> Dict[str, Any]:
        """
        Get comprehensive status of dual-mode research agent configuration.
        
        Returns:
            Status dictionary with mode, validation, and configuration details
        """
        config = SettingsRepository.get_orchestration_config()
        mode = SettingsRepository.get_research_agent_mode()
        validation = SettingsRepository.validate_research_agent_config()
        
        return {
            "current_mode": mode,
            "validation": validation,
            "has_single_config": "single_agent_config" in config,
            "has_multi_config": "multi_agent_config" in config,
            "has_legacy_config": "research_agent_model_config" in config,
            "config_keys": list(config.keys()),
            "effective_config_preview": {
                k: v for k, v in SettingsRepository.get_effective_research_config().items() 
                if k in ["parallel_agents", "task_timeout", "max_research_loops"]
            }
        }
