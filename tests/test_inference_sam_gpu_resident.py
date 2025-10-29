"""
Tests pour valider la refactorisation inference_sam.py en mode GPU-resident
"""
import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock


def test_run_segmentation_gpu_resident():
    """Test que run_segmentation() en mode GPU-resident retourne des tensors CUDA"""
    
    # Mock du modèle SAM
    mock_sam = Mock()
    mock_sam.model = Mock()
    mock_sam.model.parameters = lambda: [torch.tensor([1.0], device="cuda:0" if torch.cuda.is_available() else "cpu")]
    mock_sam.set_image = Mock()
    mock_sam.predict = Mock(return_value=(
        [np.ones((64, 64), dtype=np.float32)],  # masks
        [0.9],  # scores
        []      # low_res_masks
    ))
    
    try:
        from src.core.inference.engine.inference_sam import run_segmentation
        
        # Image d'entrée
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        bbox = np.array([10, 10, 50, 50], dtype=np.float32)
        
        # Test avec as_numpy=False (mode GPU-resident)
        result = run_segmentation(mock_sam, image, bbox, as_numpy=False)
        
        # Vérifications
        assert isinstance(result, torch.Tensor), f"Expected torch.Tensor, got {type(result)}"
        
        if torch.cuda.is_available():
            assert result.is_cuda, "Result should be on CUDA"
            
        print("✅ Test GPU-resident mode passed")
        
    except ImportError as e:
        pytest.skip(f"Cannot import run_segmentation: {e}")


def test_run_segmentation_numpy_compatibility():
    """Test que run_segmentation() en mode numpy (legacy) retourne des numpy arrays"""
    
    # Mock du modèle SAM
    mock_sam = Mock()
    mock_sam.model = Mock()
    mock_sam.model.parameters = lambda: [torch.tensor([1.0], device="cuda:0" if torch.cuda.is_available() else "cpu")]
    mock_sam.set_image = Mock()
    mock_sam.predict = Mock(return_value=(
        [np.ones((64, 64), dtype=np.float32)],  # masks
        [0.9],  # scores
        []      # low_res_masks
    ))
    
    try:
        from src.core.inference.engine.inference_sam import run_segmentation
        
        # Image d'entrée
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        bbox = np.array([10, 10, 50, 50], dtype=np.float32)
        
        # Test avec as_numpy=True (mode legacy)
        result = run_segmentation(mock_sam, image, bbox, as_numpy=True)
        
        # Vérifications
        assert isinstance(result, np.ndarray), f"Expected np.ndarray, got {type(result)}"
        assert result.dtype == bool, f"Expected bool dtype, got {result.dtype}"
        
        print("✅ Test numpy compatibility mode passed")
        
    except ImportError as e:
        pytest.skip(f"Cannot import run_segmentation: {e}")


def test_run_segmentation_default_behavior():
    """Test que le comportement par défaut (as_numpy=False) retourne des tensors"""
    
    # Mock du modèle SAM
    mock_sam = Mock()
    mock_sam.model = Mock()
    mock_sam.model.parameters = lambda: [torch.tensor([1.0], device="cuda:0" if torch.cuda.is_available() else "cpu")]
    mock_sam.set_image = Mock()
    mock_sam.predict = Mock(return_value=(
        [np.ones((64, 64), dtype=np.float32)],  # masks
        [0.9],  # scores
        []      # low_res_masks
    ))
    
    try:
        from src.core.inference.engine.inference_sam import run_segmentation
        
        # Image d'entrée
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        bbox = np.array([10, 10, 50, 50], dtype=np.float32)
        
        # Test sans spécifier as_numpy (devrait utiliser la valeur par défaut: False)
        result = run_segmentation(mock_sam, image, bbox)
        
        # Vérifications - par défaut devrait retourner des tensors
        assert isinstance(result, torch.Tensor), f"Expected torch.Tensor by default, got {type(result)}"
        
        print("✅ Test default behavior (GPU-resident) passed")
        
    except ImportError as e:
        pytest.skip(f"Cannot import run_segmentation: {e}")


def test_legacy_segmentation_gpu_resident():
    """Test _run_segmentation_legacy en mode GPU-resident"""
    
    # Mock du modèle SAM legacy
    mock_sam = Mock()
    mock_sam.parameters = lambda: [torch.tensor([1.0], device="cuda:0" if torch.cuda.is_available() else "cpu")]
    mock_sam.eval = Mock()
    
    # Mock du forward qui retourne un tensor GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    mock_outputs = [{"masks": torch.ones(1, 1, 64, 64, device=device)}]
    mock_sam.forward = Mock(return_value=mock_outputs)
    mock_sam.__call__ = mock_sam.forward
    
    try:
        from src.core.inference.engine.inference_sam import _run_segmentation_legacy
        
        # ROI d'entrée
        roi = np.zeros((64, 64), dtype=np.uint8)
        
        # Test avec as_numpy=False (mode GPU-resident)
        result = _run_segmentation_legacy(mock_sam, roi, as_numpy=False)
        
        # Vérifications
        assert isinstance(result, torch.Tensor), f"Expected torch.Tensor, got {type(result)}"
        
        if torch.cuda.is_available():
            assert result.is_cuda, "Result should be on CUDA"
            
        print("✅ Test legacy GPU-resident mode passed")
        
    except ImportError as e:
        pytest.skip(f"Cannot import _run_segmentation_legacy: {e}")


def test_legacy_segmentation_numpy_mode():
    """Test _run_segmentation_legacy en mode numpy"""
    
    # Mock du modèle SAM legacy
    mock_sam = Mock()
    mock_sam.parameters = lambda: [torch.tensor([1.0], device="cuda:0" if torch.cuda.is_available() else "cpu")]
    mock_sam.eval = Mock()
    
    # Mock du forward qui retourne un tensor GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    mock_outputs = [{"masks": torch.ones(1, 1, 64, 64, device=device)}]
    mock_sam.forward = Mock(return_value=mock_outputs)
    mock_sam.__call__ = mock_sam.forward
    
    try:
        from src.core.inference.engine.inference_sam import _run_segmentation_legacy
        
        # ROI d'entrée
        roi = np.zeros((64, 64), dtype=np.uint8)
        
        # Test avec as_numpy=True (mode legacy)
        result = _run_segmentation_legacy(mock_sam, roi, as_numpy=True)
        
        # Vérifications
        assert isinstance(result, np.ndarray), f"Expected np.ndarray, got {type(result)}"
        
        print("✅ Test legacy numpy mode passed")
        
    except ImportError as e:
        pytest.skip(f"Cannot import _run_segmentation_legacy: {e}")


@patch('src.core.inference.engine.inference_sam.time')
def test_kpi_instrumentation(mock_time):
    """Test que l'instrumentation KPI fonctionne correctement"""
    
    mock_time.time.return_value = 1234567890.0
    
    # Mock du modèle SAM legacy
    mock_sam = Mock()
    mock_sam.parameters = lambda: [torch.tensor([1.0], device="cuda:0" if torch.cuda.is_available() else "cpu")]
    mock_sam.eval = Mock()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    mock_outputs = [{"masks": torch.ones(1, 1, 64, 64, device=device)}]
    mock_sam.forward = Mock(return_value=mock_outputs)
    mock_sam.__call__ = mock_sam.forward
    
    try:
        with patch('core.monitoring.kpi.safe_log_kpi') as mock_log_kpi, \
             patch('core.monitoring.kpi.format_kpi') as mock_format_kpi:
            
            mock_format_kpi.return_value = "formatted_kpi"
            
            from src.core.inference.engine.inference_sam import _run_segmentation_legacy
            
            # ROI d'entrée
            roi = np.zeros((64, 64), dtype=np.uint8)
            
            # Test avec as_numpy=False
            _run_segmentation_legacy(mock_sam, roi, as_numpy=False)
            
            # Vérifier que les fonctions KPI ont été appelées
            mock_format_kpi.assert_called_once()
            mock_log_kpi.assert_called_once_with("formatted_kpi")
            
            # Vérifier le contenu du KPI
            call_args = mock_format_kpi.call_args[0][0]
            assert call_args["event"] == "sam_mask_output"
            assert call_args["as_numpy"] == 0  # False -> 0
            assert call_args["ts"] == 1234567890.0
            
            print("✅ Test KPI instrumentation passed")
        
    except ImportError as e:
        pytest.skip(f"Cannot import modules: {e}")


if __name__ == "__main__":
    print("🧪 Running inference_sam.py GPU-resident tests...")
    
    test_run_segmentation_gpu_resident()
    test_run_segmentation_numpy_compatibility()  
    test_run_segmentation_default_behavior()
    test_legacy_segmentation_gpu_resident()
    test_legacy_segmentation_numpy_mode()
    test_kpi_instrumentation(Mock())
    
    print("✅ All tests completed!")