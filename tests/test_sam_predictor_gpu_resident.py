"""
Tests pour valider la refactorisation SamPredictor.predict() en mode GPU-resident
"""
import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock


def test_predict_gpu_resident():
    """Test que predict() en mode GPU-resident retourne des tensors CUDA"""
    
    # Mock du modèle SAM et des composants
    mock_sam = Mock()
    mock_sam.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    mock_sam.image_encoder.img_size = 1024
    mock_sam.image_format = "RGB"
    mock_sam.mask_threshold = 0.0
    mock_sam.preprocess = lambda x: x
    mock_sam.image_encoder = lambda x: torch.randn(1, 256, 64, 64, device=mock_sam.device)
    mock_sam.prompt_encoder.get_dense_pe = lambda: torch.randn(1, 256, 64, 64, device=mock_sam.device)
    mock_sam.prompt_encoder = lambda *args, **kwargs: (
        torch.randn(1, 2, 256, device=mock_sam.device),
        torch.randn(1, 256, 64, 64, device=mock_sam.device)
    )
    mock_sam.mask_decoder = lambda *args, **kwargs: (
        torch.randn(1, 3, 256, 256, device=mock_sam.device),
        torch.randn(1, 3, device=mock_sam.device)
    )
    mock_sam.postprocess_masks = lambda low_res, input_size, orig_size: torch.randn(
        1, 3, orig_size[0], orig_size[1], device=mock_sam.device
    )
    
    # Import dynamique pour éviter les erreurs d'import
    try:
        from src.core.inference.MobileSAM.mobile_sam.predictor import SamPredictor
        
        # Créer le predictor
        predictor = SamPredictor(mock_sam)
        
        # Simuler qu'une image est set
        predictor.is_image_set = True
        predictor.features = torch.randn(1, 256, 64, 64, device=mock_sam.device)
        predictor.input_size = (1024, 1024)
        predictor.original_size = (512, 512)
        
        # Test avec as_numpy=False (mode GPU-resident)
        masks, iou, low_res = predictor.predict(as_numpy=False)
        
        # Vérifications
        assert isinstance(masks, torch.Tensor), f"Expected torch.Tensor, got {type(masks)}"
        assert isinstance(iou, torch.Tensor), f"Expected torch.Tensor, got {type(iou)}"
        assert isinstance(low_res, torch.Tensor), f"Expected torch.Tensor, got {type(low_res)}"
        
        if torch.cuda.is_available():
            assert masks.is_cuda, "Masks should be on CUDA"
            assert iou.is_cuda, "IoU should be on CUDA"
            assert low_res.is_cuda, "Low res masks should be on CUDA"
            
        print("✅ Test GPU-resident mode passed")
        
    except ImportError as e:
        pytest.skip(f"Cannot import SamPredictor: {e}")


def test_predict_numpy_compatibility():
    """Test que predict() en mode numpy (legacy) retourne des numpy arrays"""
    
    # Mock du modèle SAM
    mock_sam = Mock()
    mock_sam.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    mock_sam.image_encoder.img_size = 1024
    mock_sam.image_format = "RGB"
    mock_sam.mask_threshold = 0.0
    mock_sam.preprocess = lambda x: x
    mock_sam.image_encoder = lambda x: torch.randn(1, 256, 64, 64, device=mock_sam.device)
    mock_sam.prompt_encoder.get_dense_pe = lambda: torch.randn(1, 256, 64, 64, device=mock_sam.device)
    mock_sam.prompt_encoder = lambda *args, **kwargs: (
        torch.randn(1, 2, 256, device=mock_sam.device),
        torch.randn(1, 256, 64, 64, device=mock_sam.device)
    )
    mock_sam.mask_decoder = lambda *args, **kwargs: (
        torch.randn(1, 3, 256, 256, device=mock_sam.device),
        torch.randn(1, 3, device=mock_sam.device)
    )
    mock_sam.postprocess_masks = lambda low_res, input_size, orig_size: torch.randn(
        1, 3, orig_size[0], orig_size[1], device=mock_sam.device
    )
    
    try:
        from src.core.inference.MobileSAM.mobile_sam.predictor import SamPredictor
        
        # Créer le predictor
        predictor = SamPredictor(mock_sam)
        
        # Simuler qu'une image est set
        predictor.is_image_set = True
        predictor.features = torch.randn(1, 256, 64, 64, device=mock_sam.device)
        predictor.input_size = (1024, 1024)
        predictor.original_size = (512, 512)
        
        # Test avec as_numpy=True (mode legacy)
        masks, iou, low_res = predictor.predict(as_numpy=True)
        
        # Vérifications
        assert isinstance(masks, np.ndarray), f"Expected np.ndarray, got {type(masks)}"
        assert isinstance(iou, np.ndarray), f"Expected np.ndarray, got {type(iou)}"
        assert isinstance(low_res, np.ndarray), f"Expected np.ndarray, got {type(low_res)}"
        
        print("✅ Test numpy compatibility mode passed")
        
    except ImportError as e:
        pytest.skip(f"Cannot import SamPredictor: {e}")


def test_predict_default_behavior():
    """Test que le comportement par défaut (as_numpy=False) retourne des tensors"""
    
    # Mock du modèle SAM
    mock_sam = Mock()
    mock_sam.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    mock_sam.image_encoder.img_size = 1024
    mock_sam.image_format = "RGB"
    mock_sam.mask_threshold = 0.0
    mock_sam.preprocess = lambda x: x
    mock_sam.image_encoder = lambda x: torch.randn(1, 256, 64, 64, device=mock_sam.device)
    mock_sam.prompt_encoder.get_dense_pe = lambda: torch.randn(1, 256, 64, 64, device=mock_sam.device)
    mock_sam.prompt_encoder = lambda *args, **kwargs: (
        torch.randn(1, 2, 256, device=mock_sam.device),
        torch.randn(1, 256, 64, 64, device=mock_sam.device)
    )
    mock_sam.mask_decoder = lambda *args, **kwargs: (
        torch.randn(1, 3, 256, 256, device=mock_sam.device),
        torch.randn(1, 3, device=mock_sam.device)
    )
    mock_sam.postprocess_masks = lambda low_res, input_size, orig_size: torch.randn(
        1, 3, orig_size[0], orig_size[1], device=mock_sam.device
    )
    
    try:
        from src.core.inference.MobileSAM.mobile_sam.predictor import SamPredictor
        
        # Créer le predictor
        predictor = SamPredictor(mock_sam)
        
        # Simuler qu'une image est set
        predictor.is_image_set = True
        predictor.features = torch.randn(1, 256, 64, 64, device=mock_sam.device)
        predictor.input_size = (1024, 1024)
        predictor.original_size = (512, 512)
        
        # Test sans spécifier as_numpy (devrait utiliser la valeur par défaut: False)
        masks, iou, low_res = predictor.predict()
        
        # Vérifications - par défaut devrait retourner des tensors
        assert isinstance(masks, torch.Tensor), f"Expected torch.Tensor by default, got {type(masks)}"
        assert isinstance(iou, torch.Tensor), f"Expected torch.Tensor by default, got {type(iou)}"
        assert isinstance(low_res, torch.Tensor), f"Expected torch.Tensor by default, got {type(low_res)}"
        
        print("✅ Test default behavior (GPU-resident) passed")
        
    except ImportError as e:
        pytest.skip(f"Cannot import SamPredictor: {e}")


@patch('src.core.inference.MobileSAM.mobile_sam.predictor.time')
def test_kpi_instrumentation(mock_time):
    """Test que l'instrumentation KPI fonctionne correctement"""
    
    mock_time.time.return_value = 1234567890.0
    
    # Mock du modèle SAM
    mock_sam = Mock()
    mock_sam.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    mock_sam.image_encoder.img_size = 1024
    mock_sam.image_format = "RGB"
    mock_sam.mask_threshold = 0.0
    mock_sam.preprocess = lambda x: x
    mock_sam.image_encoder = lambda x: torch.randn(1, 256, 64, 64, device=mock_sam.device)
    mock_sam.prompt_encoder.get_dense_pe = lambda: torch.randn(1, 256, 64, 64, device=mock_sam.device)
    mock_sam.prompt_encoder = lambda *args, **kwargs: (
        torch.randn(1, 2, 256, device=mock_sam.device),
        torch.randn(1, 256, 64, 64, device=mock_sam.device)
    )
    mock_sam.mask_decoder = lambda *args, **kwargs: (
        torch.randn(1, 3, 256, 256, device=mock_sam.device),
        torch.randn(1, 3, device=mock_sam.device)
    )
    mock_sam.postprocess_masks = lambda low_res, input_size, orig_size: torch.randn(
        1, 3, orig_size[0], orig_size[1], device=mock_sam.device
    )
    
    try:
        with patch('core.monitoring.kpi.safe_log_kpi') as mock_log_kpi, \
             patch('core.monitoring.kpi.format_kpi') as mock_format_kpi:
            
            mock_format_kpi.return_value = "formatted_kpi"
            
            from src.core.inference.MobileSAM.mobile_sam.predictor import SamPredictor
            
            # Créer le predictor
            predictor = SamPredictor(mock_sam)
            
            # Simuler qu'une image est set
            predictor.is_image_set = True
            predictor.features = torch.randn(1, 256, 64, 64, device=mock_sam.device)
            predictor.input_size = (1024, 1024)
            predictor.original_size = (512, 512)
            
            # Test avec as_numpy=False
            predictor.predict(as_numpy=False)
            
            # Vérifier que les fonctions KPI ont été appelées
            mock_format_kpi.assert_called_once()
            mock_log_kpi.assert_called_once_with("formatted_kpi")
            
            # Vérifier le contenu du KPI
            call_args = mock_format_kpi.call_args[0][0]
            assert call_args["event"] == "sam_predict_output"
            assert call_args["as_numpy"] == 0  # False -> 0
            assert call_args["ts"] == 1234567890.0
            
            print("✅ Test KPI instrumentation passed")
        
    except ImportError as e:
        pytest.skip(f"Cannot import SamPredictor or KPI modules: {e}")


if __name__ == "__main__":
    print("🧪 Running SamPredictor GPU-resident tests...")
    
    test_predict_gpu_resident()
    test_predict_numpy_compatibility()  
    test_predict_default_behavior()
    test_kpi_instrumentation(Mock())
    
    print("✅ All tests completed!")