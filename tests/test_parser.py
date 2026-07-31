"""Unit tests for PrecitecData and PrecitecSurfaceAnalyzer."""
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

from precitec_data_parser import PrecitecData, PrecitecSurfaceAnalyzer


class TestPrecitecData:
    """Tests for PrecitecData class."""

    def test_imports(self):
        """Test that main classes can be imported."""
        from precitec_data_parser import PrecitecData, PrecitecSurfaceAnalyzer
        assert PrecitecData is not None
        assert PrecitecSurfaceAnalyzer is not None

    def test_unsupported_format(self):
        """Test that unsupported file formats raise NotImplementedError."""
        with pytest.raises(NotImplementedError, match="Only .csv and .bcrf exports are supported"):
            PrecitecData("test.txt")


class TestPrecitecSurfaceAnalyzer:
    """Tests for PrecitecSurfaceAnalyzer class."""

    @pytest.fixture
    def mock_precitec_data(self):
        """Create a mock PrecitecData object for testing."""
        mock_data = MagicMock(spec=PrecitecData)
        mock_data.x = np.linspace(0, 100, 50)
        mock_data.y = np.linspace(0, 100, 50)
        mock_data.z = np.random.rand(50, 50)
        mock_data.metadata = {"test": "data"}
        
        # Mock the surface
        mock_surface = MagicMock()
        mock_surface.width_um = 100.0
        mock_surface.height_um = 100.0
        mock_surface.step_x = 2.0
        mock_surface.step_y = 2.0
        mock_data.to_surface.return_value = mock_surface
        
        return mock_data

    def test_analyzer_initialization(self, mock_precitec_data):
        """Test that analyzer can be initialized with PrecitecData."""
        analyzer = PrecitecSurfaceAnalyzer(mock_precitec_data)
        assert analyzer.data == mock_precitec_data
        assert analyzer.surface is not None

    def test_filter_profile_method_validation(self, mock_precitec_data):
        """Test that invalid filter methods raise ValueError."""
        analyzer = PrecitecSurfaceAnalyzer(mock_precitec_data)
        mock_profile = MagicMock()
        
        with pytest.raises(ValueError, match='Unknown filter method'):
            analyzer.filter_profile(mock_profile, method="invalid_method")


class TestPackageVersion:
    """Tests for package version."""

    def test_version_exists(self):
        """Test that __version__ is defined."""
        import precitec_data_parser
        assert hasattr(precitec_data_parser, '__version__')
        assert isinstance(precitec_data_parser.__version__, str)
        )
        analyzer.plot_profile(profile_oblique_offset, show=True)

